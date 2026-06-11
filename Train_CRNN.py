# train_crnn_ctc_beam_fixed_improved.py  (drop-in replacement)
import os, cv2, csv, datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# =====================
# CONFIG
# =====================
DATASET_PATH = "/mnt/c/Study/study/Minor 1/Final/iam_words"
WORDS_TXT = os.path.join(DATASET_PATH, "words.txt")
IMG_DIR = os.path.join(DATASET_PATH, "words")
IMG_HEIGHT, IMG_WIDTH = 32, 128
BATCH_SIZE, EPOCHS = 32, 30     # <<< CHANGED: smaller batch, more epochs
LR = 1e-4                       # raise from 1e-6
OUT_DIR = "./out_crnn_ctc_beam"
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_FILE = os.path.join(OUT_DIR, f"iam_cache_{IMG_HEIGHT}x{IMG_WIDTH}.npz")
CKPT = os.path.join(OUT_DIR, "best_ctc_model.h5")
FINAL = os.path.join(OUT_DIR, "final_crnn_ctc_model.keras")
LOSS_PNG = os.path.join(OUT_DIR, "training_loss.png")
PRED_CSV = os.path.join(OUT_DIR, "predictions.csv")

print("🚀 CRNN+CTC Fine-Tune with Beam Search (improved)")

# =====================
# GPU / precision
# =====================
try:
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)
except Exception: pass
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# =====================
# Data helpers
# =====================
def get_img_path(img_id):
    p = img_id.split("-")
    return os.path.join(IMG_DIR, p[0], p[0]+"-"+p[1], img_id+".png")

def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    h = IMG_HEIGHT
    scale = h / img.shape[0]
    new_w = max(1, int(img.shape[1]*scale))
    img = cv2.resize(img, (min(new_w, IMG_WIDTH), h))
    if new_w < IMG_WIDTH:
        img = np.pad(img, ((0,0),(0,IMG_WIDTH-new_w)), mode="constant", constant_values=255)
    elif new_w > IMG_WIDTH:
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = (255.0 - img).astype("float32")/255.0
    return np.expand_dims(img,-1)

def load_labels(txt):
    lbl = {}
    with open(txt,"r",encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#"): continue
            p = ln.strip().split()
            if len(p)>=9 and p[1]=="ok": lbl[p[0]]=p[-1]
    return lbl

# =====================
# Cache or build
# =====================
if os.path.exists(CACHE_FILE):
    c = np.load(CACHE_FILE, allow_pickle=True)
    X, texts = c["X"], c["texts"].tolist()
else:
    labels = load_labels(WORDS_TXT)
    imgs, texts = [], []
    for k in tqdm(labels.keys(), desc="Preprocess"):
        p = get_img_path(k)
        im = preprocess(p)
        if im is not None:
            imgs.append(im); texts.append(labels[k])
    X = np.array(imgs,dtype=np.float32)
    np.savez_compressed(CACHE_FILE,X=X,texts=np.array(texts,dtype=object))
print(f"Loaded {len(X)} samples.")

# =====================
# Encode labels
# =====================
charset = sorted(set(c for s in texts for c in s))
char_to_idx = {c:i+1 for i,c in enumerate(charset)}
idx_to_char = {i+1:c for i,c in enumerate(charset)}
max_len = max(len(s) for s in texts)
num_classes = len(charset)+1  # +blank

y = np.zeros((len(texts),max_len),dtype=np.int32)
for i,s in enumerate(texts):
    for j,ch in enumerate(s[:max_len]):
        y[i,j]=char_to_idx.get(ch,0)

perm=np.random.permutation(len(X)); X=X[perm]; y=y[perm]
split=int(0.9*len(X))
Xtr,Xv=X[:split],X[split:]; ytr,yv=y[:split],y[split:]
print("Train:",Xtr.shape," Val:",Xv.shape)

# =====================
# Augmentation (NEW) <<< CHANGED
# =====================
def augment_image(img):
    # img float32 HxWx1 in [0,1]
    h,w,_ = img.shape
    im = img[:,:,0].copy()
    # horizontal scaling (width jitter)
    if random.random() < 0.5:
        scale = random.uniform(0.92,1.08)
        new_w = max(1, int(w*scale))
        im = cv2.resize(im, (new_w, h))
        if new_w < w:
            im = np.pad(im, ((0,0),(0,w-new_w)), mode="constant", constant_values=0.0)
        else:
            im = cv2.resize(im, (w,h))
    # small translations
    if random.random() < 0.5:
        tx = random.randint(-3,3)
        ty = random.randint(-1,1)
        M = np.float32([[1,0,tx],[0,1,ty]])
        im = cv2.warpAffine((im*255).astype(np.uint8), M, (w,h), borderValue=0)
        im = im.astype("float32")/255.0
    # brightness/contrast
    if random.random() < 0.3:
        alpha = random.uniform(0.95,1.08)
        im = np.clip(im * alpha, 0, 1)
    # gaussian noise
    if random.random() < 0.25:
        noise = np.random.normal(0, 0.02, im.shape).astype("float32")
        im = np.clip(im + noise, 0, 1)
    return np.expand_dims(im, -1)

def data_generator(X, y, batch_size, augment=False):
    n = len(X)
    idxs = np.arange(n)
    while True:
        np.random.shuffle(idxs)
        for i in range(0,n,batch_size):
            batch_idx = idxs[i:i+batch_size]
            XB = X[batch_idx].copy()
            yB = y[batch_idx]
            if augment:
                for k in range(len(XB)):
                    if random.random() < 0.8:
                        XB[k] = augment_image(XB[k])
            tin = np.ones((len(XB),1),dtype=np.int32)*(IMG_WIDTH//4)
            tlen = np.array([[int((r!=0).sum())] for r in yB],dtype=np.int32)
            y0 = np.zeros((len(XB),1))
            yield [XB,yB,tin,tlen], y0

# =====================
# Model (small reg changes) <<< CHANGED
# =====================
def build(input_shape,nclass):
    inp=layers.Input(shape=input_shape)
    x=layers.Conv2D(64,3,padding="same",activation="relu")(inp); x=layers.MaxPooling2D((2,2))(x)
    x=layers.Dropout(0.1)(x)                 # <<< CHANGED
    x=layers.Conv2D(128,3,padding="same",activation="relu")(x); x=layers.MaxPooling2D((2,2))(x)
    x=layers.Dropout(0.1)(x)                 # <<< CHANGED
    x=layers.Conv2D(256,3,padding="same",activation="relu")(x); x=layers.BatchNormalization()(x); x=layers.MaxPooling2D((2,1))(x)
    x=layers.Dropout(0.1)(x)                 # <<< CHANGED
    x=layers.Conv2D(512,3,padding="same",activation="relu")(x); x=layers.BatchNormalization()(x); x=layers.MaxPooling2D((2,1))(x)
    x=layers.Reshape((IMG_WIDTH//4,-1))(x)
    x=layers.Bidirectional(layers.LSTM(256,return_sequences=True,recurrent_dropout=0.12))(x)  # <<< CHANGED
    x=layers.Bidirectional(layers.LSTM(256,return_sequences=True,recurrent_dropout=0.12))(x)  # <<< CHANGED
    out=layers.Dense(nclass,activation=None,dtype="float32")(x)  # logits!
    return models.Model(inp,out,name="crnn")

crnn=build((IMG_HEIGHT,IMG_WIDTH,1),num_classes)
crnn.summary()

# =====================
# CTC loss
# =====================
labels_in=layers.Input(shape=(max_len,),dtype="int32")
inp_len=layers.Input(shape=(1,),dtype="int32")
lab_len=layers.Input(shape=(1,),dtype="int32")
def ctc_func(args):
    y_pred,lab,inp_l,lab_l=args
    lab=tf.cast(lab,tf.int32)
    nz=tf.where(lab!=0); vals=tf.gather_nd(lab,nz)-1
    sparse=tf.SparseTensor(indices=nz,values=vals,dense_shape=tf.cast(tf.shape(lab),tf.int64))
    inp_l=tf.cast(tf.reshape(inp_l,[-1]),tf.int32)
    lab_l=tf.cast(tf.reshape(lab_l,[-1]),tf.int32)
    loss=tf.nn.ctc_loss(labels=sparse,logits=y_pred,label_length=lab_l,logit_length=inp_l,
                        logits_time_major=False,blank_index=num_classes-1)
    return tf.expand_dims(loss,1)

loss_out=layers.Lambda(ctc_func,name="ctc")([crnn.output,labels_in,inp_len,lab_len])
model=models.Model([crnn.input,labels_in,inp_len,lab_len],loss_out)
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss=lambda y_true,y_pred:y_pred)
print("Model compiled with CTC loss.")

# =====================
# Generators and steps <<< CHANGED
# =====================
train_gen = data_generator(Xtr,ytr,BATCH_SIZE,augment=True)
val_gen = data_generator(Xv,yv,BATCH_SIZE,augment=False)
steps_per_epoch = max(1, len(Xtr)//BATCH_SIZE)
val_steps = max(1, len(Xv)//BATCH_SIZE)

# =====================
# ValWordAcc callback (NEW) <<< CHANGED
# =====================
class ValWordAccCallback(tf.keras.callbacks.Callback):
    def __init__(self, crnn_model, Xv, yv_array, idx_to_char, save_path, sample_n=200):
        super().__init__()
        self.crnn = crnn_model
        self.Xv = Xv
        self.yv_array = yv_array
        self.idx_to_char = idx_to_char
        self.best_acc = -1.0
        self.save_path = save_path
        self.sample_n = min(sample_n, len(Xv))

    def ctc_beam_decode(self, logits, beam=5):
        logits_tf = tf.convert_to_tensor(logits)
        time_major = tf.transpose(logits_tf, [1,0,2])
        seqs, _ = tf.nn.ctc_beam_search_decoder(inputs=time_major,
                                                sequence_length=tf.cast(tf.fill([tf.shape(logits_tf)[0]], tf.shape(logits_tf)[1]), tf.int32),
                                                beam_width=beam)
        dense = tf.sparse.to_dense(seqs[0], default_value=-1).numpy()
        dec=[]
        for row in dense:
            s=[]
            for idx in row:
                if int(idx) >= 0:
                    s.append(self.idx_to_char.get(int(idx)+1,""))
            dec.append("".join(s))
        return dec

    def on_epoch_end(self, epoch, logs=None):
        idx = np.random.choice(len(self.Xv), self.sample_n, replace=False)
        Xs = self.Xv[idx]
        logits = self.crnn.predict(Xs, batch_size=64)
        preds = self.ctc_beam_decode(logits, beam=5)
        gts = []
        for i in idx:
            g = "".join([idx_to_char.get(int(x),"") for x in yv[i] if x>0])
            gts.append(g)
        wc = sum(1 for a,b in zip(preds,gts) if a==b)
        acc = wc/len(gts)*100
        print(f"\n[ValWordAcc] epoch={epoch+1}  WordAcc={acc:.2f}%  ({wc}/{len(gts)})")
        if acc > self.best_acc:
            self.best_acc = acc
            print("Saving best model by WordAcc ->", self.save_path)
            crnn.save_weights(self.save_path)

val_acc_cb = ValWordAccCallback(crnn, Xv, yv, idx_to_char, CKPT, sample_n=200)

# =====================
# Callbacks (added earlystop + ValWordAcc) <<< CHANGED
# =====================
cb=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=False, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=3,min_lr=1e-6,verbose=1),
    tf.keras.callbacks.ModelCheckpoint(CKPT,monitor="val_loss",save_best_only=True,verbose=1),
    val_acc_cb]

# =====================
# Train (using generators) <<< CHANGED
# =====================
hist=model.fit(train_gen, validation_data=val_gen,
               steps_per_epoch=steps_per_epoch, validation_steps=val_steps,
               epochs=EPOCHS, callbacks=cb, verbose=1)

# save final model & loss plot (unchanged)
crnn.save(FINAL)
print("Saved:",FINAL)

plt.figure(figsize=(8,5))
plt.plot(hist.history["loss"]); plt.plot(hist.history["val_loss"])
plt.legend(["train","val"]); plt.title("CTC Loss"); plt.grid(True)
plt.savefig(LOSS_PNG); print("Saved plot:",LOSS_PNG)

# =====================
# Beam search decode (unchanged)
# =====================
def beam_decode(logits, beam=10):
    """
    logits: numpy array (N, T, C) - model output (logits, batch-major)
    returns: list of decoded strings (length N)
    """
    logits_tf = tf.convert_to_tensor(logits)  # shape (N, T, C)
    time_major = tf.transpose(logits_tf, [1, 0, 2])  # (T, N, C)

    seqs, _ = tf.nn.ctc_beam_search_decoder(
        inputs=time_major,
        sequence_length=tf.cast(tf.fill([tf.shape(logits_tf)[0]], tf.shape(logits_tf)[1]), tf.int32),
        beam_width=beam
    )
    dense_decoded = tf.sparse.to_dense(seqs[0], default_value=-1).numpy()  # shape (N, max_decoded_len)

    decoded_texts = []
    for row in dense_decoded:
        chars = []
        for idx in row:
            if int(idx) >= 0:
                chars.append(idx_to_char.get(int(idx) + 1, ""))
        decoded_texts.append("".join(chars))
    return decoded_texts

def edit_dist(a,b):
    la,lb=len(a),len(b); dp=[[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): dp[i][0]=i
    for j in range(lb+1): dp[0][j]=j
    for i in range(1,la+1):
        for j in range(1,lb+1):
            c=0 if a[i-1]==b[j-1] else 1
            dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+c)
    return dp[la][lb]

if os.path.exists(CKPT):
    try:
        crnn.load_weights(CKPT)
        print("Loaded best weights for decode.")
    except Exception:
        pass

N=min(1000,len(Xv))
logits=crnn.predict(Xv[:N],batch_size=BATCH_SIZE)
preds=beam_decode(logits,beam=10)

gt=[]
for seq in yv[:N]:
    gt.append("".join([idx_to_char.get(int(x),"") for x in seq if x>0]))

err=tot=wc=0
for p,g in zip(preds,gt):
    err+=edit_dist(p,g); tot+=max(1,len(g))
    if p==g: wc+=1
cer=err/tot; acc=wc/len(gt)*100
print(f"CER={cer*100:.2f}%  WordAcc={acc:.2f}%  ({wc}/{len(gt)})")

with open(PRED_CSV,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["i","gt","pred"])
    for i,(g,p) in enumerate(zip(gt,preds)): w.writerow([i,g,p])
print("Saved CSV:",PRED_CSV)
