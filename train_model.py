"""
Mental Health Sentiment RNN - Model Retraining Script
Uses real-world CARER emotion dataset + targeted synthetic data
to train a robust RNN classifier across 7 mental health classes.
"""

import re
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, GlobalMaxPooling1D, Dense, Dropout

# -------------------------------------------------------------------
# REPRODUCIBILITY
# -------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print(" Mental Health Sentiment AI - Model Training Pipeline")
print("=" * 60)

# -------------------------------------------------------------------
# STEP 1: DOWNLOAD REAL-WORLD EMOTION DATASET (CARER / dair-ai)
# These are genuine tweets/social media texts - rich, diverse vocab
# -------------------------------------------------------------------
print("\n[1/7] Downloading real-world CARER emotion dataset...")

TRAIN_URL = "https://raw.githubusercontent.com/ataislucky/Data-Science/main/dataset/emotion_train.txt"
VAL_URL   = "https://raw.githubusercontent.com/ataislucky/Data-Science/main/dataset/emotion_val.txt"
TEST_URL  = "https://raw.githubusercontent.com/ataislucky/Data-Science/main/dataset/emotion_test.txt"

dfs = []
for url in [TRAIN_URL, VAL_URL, TEST_URL]:
    try:
        df_part = pd.read_csv(url, sep=';', names=['text', 'emotion'])
        dfs.append(df_part)
        print(f"  ✓ Loaded {len(df_part)} rows from {url.split('/')[-1]}")
    except Exception as e:
        print(f"  ✗ Failed to load {url}: {e}")

real_df = pd.concat(dfs, ignore_index=True)
print(f"  Total real samples: {len(real_df)}")
print(f"  Real classes found: {real_df['emotion'].value_counts().to_dict()}")

# -------------------------------------------------------------------
# STEP 2: MAP REAL LABELS → OUR 7 APP CLASSES
# Mapping: joy→Happy, sadness→Sadness, anger→Anger,
#          fear→Anxiety, love→Happy, surprise→Panic (closest match)
# -------------------------------------------------------------------
print("\n[2/7] Mapping real labels to our 7 target classes...")

label_map = {
    'joy':      'Happy',
    'love':     'Happy',
    'sadness':  'Sadness',
    'anger':    'Anger',
    'fear':     'Anxiety',
    'surprise': 'Panic',
}
real_df['label'] = real_df['emotion'].map(label_map)
real_df = real_df[real_df['label'].notna()][['text', 'label']].copy()
print(f"  Mapped class distribution: {real_df['label'].value_counts().to_dict()}")

# -------------------------------------------------------------------
# STEP 3: SYNTHETIC DATA FOR MISSING CLASSES (Depression, Stress, Panic boost)
# We write rich, varied sentences using diverse natural vocabulary
# -------------------------------------------------------------------
print("\n[3/7] Generating supplementary synthetic data for missing classes...")

depression_sentences = [
    "I feel completely empty inside, like nothing matters anymore.",
    "There is no point to anything, I have lost all hope.",
    "I cannot get out of bed, the darkness is overwhelming.",
    "I feel worthless and like a burden to everyone around me.",
    "Nothing brings me joy anymore, everything feels hollow.",
    "I am so numb, I cannot feel anything at all.",
    "I feel utterly hopeless about my future and my life.",
    "I have no energy, no motivation, and no will to go on.",
    "Everything feels pointless and dark and empty.",
    "I am drowning in sadness and despair that never goes away.",
    "I feel like disappearing, like no one would even notice.",
    "My mind is a fog of misery and I cannot see any light.",
    "I have stopped caring about myself or the people around me.",
    "I feel isolated and broken, completely shut off from the world.",
    "The heaviness in my chest never goes away, I feel so depressed.",
    "I wake up every day feeling hopeless and exhausted.",
    "I cannot stop thinking about how worthless I am.",
    "I feel so lost and alone, like no one understands my pain.",
    "Every day feels like a struggle just to survive and breathe.",
    "I have been crying for days and I do not even know why.",
    "The emptiness inside me is consuming everything I have left.",
    "I feel deeply depressed and cannot find any reason to smile.",
    "My world is completely grey and nothing feels meaningful anymore.",
    "I am sinking deeper into hopelessness and I cannot stop it.",
    "I feel like a ghost just going through the motions of life.",
    "All I feel is a deep, overwhelming sadness that never lifts.",
    "I have given up on trying, nothing ever changes for me.",
    "I feel crushed under the weight of my own despair.",
    "I am lost in a dark fog and I cannot find my way out.",
    "I feel so disconnected from life, like I am watching from afar.",
    "Nothing excites me anymore, I feel completely dead inside.",
    "I am struggling to find any reason to keep going.",
    "I feel mentally exhausted and emotionally drained every single day.",
    "The sadness never leaves me, it follows me everywhere I go.",
    "I feel like I am trapped in a pit with no way out.",
    "I have lost all interest in things that used to make me happy.",
    "I feel so broken that I do not know if I can be fixed.",
    "My heart feels heavy with grief and sorrow all the time.",
    "I feel like a failure in every possible aspect of my life.",
    "I cannot remember the last time I felt truly happy or okay.",
    "I feel as though the world would be better without me in it.",
    "Depression has stolen my ability to feel anything positive.",
    "I am consumed by darkness and I cannot find any hope.",
    "I feel so alone even when I am surrounded by other people.",
    "My emotional pain is unbearable and I do not know what to do.",
    "I feel mentally shattered and I see no path forward.",
    "I am deeply depressed and struggling to cope with daily life.",
    "The weight of my sadness makes it hard to even breathe.",
    "I feel completely helpless and unable to change my situation.",
    "I am in so much emotional pain and I have nowhere to turn.",
]

stress_sentences = [
    "I am completely overwhelmed with all the work I have to do.",
    "These deadlines are crushing me and I cannot keep up anymore.",
    "I have so much on my plate and I cannot handle it all.",
    "I am burning out from the constant pressure at work.",
    "My workload is impossible and I feel stretched to the limit.",
    "I have no time to rest, I am exhausted from all the stress.",
    "The pressure is enormous and I feel like I am about to collapse.",
    "I cannot cope with everything that is being demanded from me.",
    "I am running on empty and the stress is killing me slowly.",
    "Everything is piling up and I feel completely crushed by it.",
    "I have so many responsibilities and I cannot manage them all.",
    "The never-ending to-do list is making me lose my mind.",
    "I am totally overwhelmed and stressed out beyond my limits.",
    "Work pressure is unbearable and I feel I cannot breathe.",
    "I am so stressed that I cannot sleep or eat properly.",
    "My boss keeps piling on more tasks and I am reaching my breaking point.",
    "I feel so pressured and overworked, I cannot take it anymore.",
    "Every single day is a battle against an endless pile of tasks.",
    "I am mentally drained from the constant demands at work and home.",
    "The stress in my life has become completely unmanageable.",
    "I feel overwhelmed by all my responsibilities and obligations.",
    "I am stressed beyond measure and cannot find any relief.",
    "Too many deadlines, too much work, and too little time.",
    "I cannot keep up with everything and it is making me fall apart.",
    "The constant pressure makes me feel like I am going to break down.",
    "I am exhausted from being stressed out every single day.",
    "I have no time for myself because work takes everything from me.",
    "All this pressure has made my body tense and my mind foggy.",
    "I am so stressed that even small tasks feel impossible to handle.",
    "The weight of my responsibilities is crushing my mental health.",
    "I feel completely burnt out and unable to function anymore.",
    "Work is consuming my life and I am so overwhelmed.",
    "I keep making mistakes because I am too stressed to think straight.",
    "There is so much pressure on me and I cannot cope.",
    "I feel mentally exhausted from dealing with constant stress.",
    "My stress levels are through the roof and I need a break.",
    "I am overwhelmed with assignments, deadlines, and responsibilities.",
    "The pressure is relentless and I feel like I might snap.",
    "I cannot relax even for a moment because there is always more to do.",
    "I am under so much stress that my physical health is suffering too.",
    "Everything is too much and I am totally overwhelmed.",
    "I feel like I am drowning in responsibilities and there is no escape.",
    "I am so stressed about my exams that I cannot concentrate at all.",
    "My mind is racing from all the tasks I have not completed yet.",
    "I feel maxed out and stressed to the absolute breaking point.",
    "The demands on my time and energy are completely unsustainable.",
    "I am so overstretched that I feel close to a complete breakdown.",
    "All this stress is making me irritable, exhausted, and hopeless.",
    "I am buried under work and I cannot see any end in sight.",
    "I feel like I cannot breathe from all the pressure surrounding me.",
]

panic_sentences = [
    "My heart is pounding so fast, I cannot breathe at all.",
    "I feel like I am dying right now, everything is spinning around me.",
    "I had a sudden panic attack and I completely lost control.",
    "I feel a terrifying rush of fear and my body is trembling.",
    "I cannot catch my breath, I am hyperventilating and panicking.",
    "My chest is so tight I feel like I am having a heart attack.",
    "I am in a complete panic and cannot calm myself down.",
    "I feel terrified and suffocated and cannot escape this feeling.",
    "Suddenly I felt an overwhelming sense of terror and dread.",
    "I started shaking and sweating uncontrollably from panic.",
    "I felt trapped with no way out and my panic escalated rapidly.",
    "I cannot control my breathing and I am absolutely terrified.",
    "The panic hits me suddenly and without any warning at all.",
    "I feel completely overwhelmed by an intense wave of terror.",
    "My hands were shaking and I could not think clearly from the panic.",
    "I feel like everything is closing in on me and I cannot escape.",
    "The fear hit me so fast and so hard I thought I was dying.",
    "I am in a complete state of panic and cannot function at all.",
    "I screamed because the panic came so suddenly and intensely.",
    "My body is in full fight or flight mode and I feel terrified.",
    "I feel an unbearable sense of dread and my heart is racing wildly.",
    "The panic attack came out of nowhere and left me shaking.",
    "I am scared out of my mind and my whole body is frozen with terror.",
    "I cannot calm down, the panic is overwhelming my entire body.",
    "I feel like I might faint from the intensity of this panic attack.",
    "I am paralyzed with terror and cannot think or move rationally.",
    "Panic overwhelmed me completely and I could not breathe at all.",
    "My mind is spinning with irrational fear I cannot control.",
    "I felt an explosive rush of panic and terror take over my body.",
    "I am gripped by intense, uncontrollable fear and dread.",
    "My heart is racing out of control and I am absolutely terrified.",
    "I am in the middle of a panic attack and I feel like I will die.",
    "I cannot stop the panic, it is consuming my entire mind and body.",
    "The terror is so intense that I cannot think or breathe properly.",
    "I am overcome with a sudden wave of panic and pure terror.",
    "I feel completely unhinged by the severity of this panic attack.",
    "I am gasping for air and shaking from an intense panic episode.",
    "Sudden terror gripped me and I lost all control of my senses.",
    "I am overwhelmed with fear and my body is completely rigid.",
    "The panic attack made me feel like I was completely losing my mind.",
    "I can feel my heart pounding in my throat from sheer terror.",
    "I am consumed by panic and I cannot find any sense of calm.",
    "I feel like I am spinning out of control and cannot stop the panic.",
    "My breathing is rapid and shallow because of this terrifying panic.",
    "I am in a blind panic and everything around me feels unreal.",
    "I felt a sudden overwhelming terror that stopped me in my tracks.",
    "This panic attack is the worst feeling I have ever experienced.",
    "I am completely frozen with fear and cannot move or think clearly.",
    "I cannot escape the horrifying feeling that something terrible will happen.",
    "The panic has completely consumed my ability to function or think.",
]

synthetic_rows = []
for sentence in depression_sentences:
    synthetic_rows.append({'text': sentence, 'label': 'Depression'})
for sentence in stress_sentences:
    synthetic_rows.append({'text': sentence, 'label': 'Stress'})
for sentence in panic_sentences:
    synthetic_rows.append({'text': sentence, 'label': 'Panic'})

synth_df = pd.DataFrame(synthetic_rows)
print(f"  Synthetic samples added: {synth_df['label'].value_counts().to_dict()}")

# -------------------------------------------------------------------
# STEP 4: MERGE AND BALANCE THE DATASET
# -------------------------------------------------------------------
print("\n[4/7] Merging and balancing the full dataset...")

full_df = pd.concat([real_df, synth_df], ignore_index=True)
full_df = full_df.dropna(subset=['text', 'label'])
full_df['text'] = full_df['text'].astype(str)

print(f"  Pre-balance distribution:\n{full_df['label'].value_counts()}")

# Balance all classes to 2000 samples each using oversampling
target_count = 2000
balanced_parts = []
for cls in full_df['label'].unique():
    cls_df = full_df[full_df['label'] == cls]
    if len(cls_df) >= target_count:
        balanced_parts.append(cls_df.sample(n=target_count, random_state=42))
    else:
        balanced_parts.append(resample(cls_df, n_samples=target_count, random_state=42))

balanced_df = pd.concat(balanced_parts, ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  Final balanced distribution:\n{balanced_df['label'].value_counts()}")
print(f"  Total samples: {len(balanced_df)}")

# -------------------------------------------------------------------
# STEP 5: CLEAN TEXT + FIT TOKENIZER & LABEL ENCODER
# -------------------------------------------------------------------
print("\n[5/7] Preprocessing text and fitting tokenizer...")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

balanced_df['clean_text'] = balanced_df['text'].apply(clean_text)

VOCAB_SIZE  = 10000
MAX_LENGTH  = 100

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(balanced_df['clean_text'])

label_encoder = LabelEncoder()
balanced_df['encoded'] = label_encoder.fit_transform(balanced_df['label'])

print(f"  Actual vocabulary size: {len(tokenizer.word_index)}")
print(f"  Label classes: {list(label_encoder.classes_)}")

# Save artifacts
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("  ✓ tokenizer.pkl and label_encoder.pkl saved.")

# -------------------------------------------------------------------
# STEP 6: PREPARE SEQUENCES AND TRAIN/VAL SPLIT
# -------------------------------------------------------------------
print("\n[6/7] Preparing sequences...")

sequences = tokenizer.texts_to_sequences(balanced_df['clean_text'])
padded    = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
labels    = balanced_df['encoded'].values

X_train, X_val, y_train, y_val = train_test_split(
    padded, labels, test_size=0.15, random_state=42, stratify=labels
)
print(f"  Train set: {X_train.shape}, Validation set: {X_val.shape}")

# -------------------------------------------------------------------
# STEP 7: BUILD AND TRAIN THE SIMPLERNN MODEL
# -------------------------------------------------------------------
print("\n[7/7] Building and training the SimpleRNN model...")

num_classes = len(label_encoder.classes_)
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE+1, output_dim=128),
    SimpleRNN(128, return_sequences=True, activation='tanh'),
    SimpleRNN(64, return_sequences=True, activation='tanh'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Explicitly build the model to initialize shapes
model.build(input_shape=(None, MAX_LENGTH))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Show model summary after building
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=callbacks
)

# Save model
model.save('upgraded_RNN_model.h5')
print("\n✓ Model saved to upgraded_RNN_model.h5")

# -------------------------------------------------------------------
# QUICK VERIFICATION
# -------------------------------------------------------------------
print("\n--- Quick Verification ---")
test_sentences = [
    ("I feel completely empty and hopeless, nothing matters anymore",          "Depression"),
    ("I am so stressed and overwhelmed with all this work and deadlines",      "Stress"),
    ("My heart is pounding and I cannot breathe, I am in a panic",            "Panic"),
    ("I am so angry and furious about what just happened to me",               "Anger"),
    ("I feel so sad and heartbroken, I have been crying all day long",         "Sadness"),
    ("I am very worried and anxious about my future and what will happen",     "Anxiety"),
    ("I feel so happy and excited about this wonderful new opportunity",        "Happy"),
]

for text, expected in test_sentences:
    cleaned  = clean_text(text)
    seq      = tokenizer.texts_to_sequences([cleaned])
    padded_s = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    probs    = model.predict(padded_s, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_lbl = label_encoder.classes_[pred_idx]
    conf     = probs[pred_idx] * 100
    status   = "✅" if pred_lbl == expected else "❌"
    print(f"  {status} Expected={expected:12s} | Got={pred_lbl:12s} ({conf:5.1f}%) | '{text[:55]}...'")

print("\n=== Training Complete! Run: streamlit run app.py ===")
