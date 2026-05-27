import re
import pickle
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, GlobalMaxPooling1D, Dropout

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("Starting Mental Health Emotion Model Retraining Pipeline...")

# =====================================================================
# STEP 1: VOCABULARY AND TEMPLATES DEFINITION FOR CORPUS GENERATION
# =====================================================================
synonyms = {
    "Anger": ["angry", "frustrated", "furious", "irritated", "annoyed", "mad", "ticked off", "pissed", "enraged", "resentful", "outraged", "hostile", "bitter", "fuming", "agitated", "exasperated", "vexed"],
    "Anxiety": ["anxious", "worried", "nervous", "tense", "uneasy", "apprehensive", "jittery", "restless", "fearful", "paranoid", "shaky", "dreadful", "timid", "fretful"],
    "Depression": ["depressed", "hopeless", "empty", "hollow", "worthless", "miserable", "despondent", "heavy", "gloomy", "dejected", "bleak", "numb", "despairing", "downcast", "melancholy"],
    "Happy": ["happy", "excited", "joyful", "peaceful", "content", "cheerful", "delighted", "thrilled", "ecstatic", "grateful", "optimistic", "glad", "wonderful", "satisfied", "radiant", "blissful", "elated"],
    "Panic": ["panicked", "terrified", "suffocating", "choking", "trapped", "alarmed", "hysterical", "frightened", "horrified", "frantic", "shaken"],
    "Sadness": ["sad", "sorrowful", "devastated", "blue", "down", "heartbroken", "gloomy", "unhappy", "dejected", "grieving", "tearful", "mournful", "weeping", "crestfallen"],
    "Stress": ["stressed", "overwhelmed", "overloaded", "pressured", "burnt out", "exhausted", "tired", "drained", "stretched", "weary", "strained"]
}

contexts = [
    "my job", "my future", "my exams", "what they said", "my family", "school", "work", 
    "deadlines", "my relationship", "life", "everything", "nothing", "my finances", 
    "my health", "the upcoming presentation", "the news", "society", "the situation", "my career"
]

intensifiers = [
    "extremely", "very", "completely", "really", "incredibly", "so", "terribly", 
    "unbelievably", "totally", "deeply", "highly", "absolutely", "so damn", "rather", "pretty"
]

subjects = [
    "I am", "I feel", "I'm", "She is", "She feels", "He is", "He feels", 
    "They are", "They feel", "We are", "We feel", "My friend is", "My friend feels", 
    "Everyone is", "People are"
]

templates = {
    "Anger": [
        "{subj} {intensifier} {adj} about {context}.",
        "{subj} feeling so {adj} because of {context}.",
        "This is so unfair, it makes me {intensifier} {adj}.",
        "I can't stand {context} anymore, I feel {adj}!",
        "Why does {context} always make me so {adj}?",
        "{subj} absolutely {adj} and boiling with rage.",
        "This situation makes my blood boil, I am so {adj}.",
        "{subj} {adj} and irritated by {context}.",
        "I want to scream, this is {intensifier} {adj}.",
        "{subj} completely fed up and {adj}.",
        "I hate {context}, it makes me feel {intensifier} {adj}.",
        "{subj} {intensifier} {adj} and furious at {context}.",
        "{subj} {intensifier} {adj} with how things are going.",
        "I feel extremely {adj} and annoyed right now."
    ],
    "Anxiety": [
        "{subj} {intensifier} {adj} about {context}.",
        "I can't stop overthinking {context}, I'm so {adj}.",
        "My heart is racing and I feel {intensifier} {adj}.",
        "{subj} feeling so {adj} about the upcoming presentation.",
        "{subj} extremely nervous and {adj} about {context}.",
        "My hands are shaking, I feel so {adj}.",
        "{subj} {intensifier} {adj} and I can't seem to calm down.",
        "{subj} {intensifier} {adj} about my future and career.",
        "I'm constantly worrying and feeling {adj} about {context}.",
        "{subj} {adj} and restless every single day.",
        "{subj} {adj} and worried about {context}.",
        "{subj} {intensifier} {adj} and tense today.",
        "I'm feeling incredibly {adj} and uneasy.",
        "My mind is constantly racing, making me feel so {adj}.",
        "I feel so {adj} and nervous about what will happen."
    ],
    "Depression": [
        "{subj} {intensifier} {adj} and empty inside.",
        "I can't even get out of bed, I'm so {adj}.",
        "Everything feels completely dark, I'm {intensifier} {adj}.",
        "{subj} feeling like a burden and so {adj} today.",
        "Nothing matters anymore, I feel {intensifier} {adj}.",
        "{subj} completely numb and {adj} inside.",
        "What's the point? I feel so {adj} and lonely.",
        "I feel a heavy weight on my chest, I am so {adj}.",
        "My energy is entirely gone, I feel {intensifier} {adj}.",
        "{subj} so {adj} and isolated from everyone.",
        "{subj} {intensifier} {adj} and has lost all hope.",
        "{subj} incredibly {adj} and miserable.",
        "Everything is bleak, I am so {adj} and exhausted.",
        "I feel so lonely, sad, and {adj}."
    ],
    "Happy": [
        "{subj} {intensifier} {adj} about {context}!",
        "Today was a wonderful day, I feel so {adj}.",
        "{subj} so excited and {adj} for the future.",
        "{subj} {intensifier} {adj}, peaceful, and content.",
        "Everything is going perfectly, I am so {adj}!",
        "I can't stop smiling, I feel {intensifier} {adj}.",
        "I feel incredibly blessed, joyful, and {adj}.",
        "I'm {intensifier} {adj} to start this new journey.",
        "It is a beautiful day, I feel so {adj} and relaxed.",
        "{subj} very positive, resilient, and {adj}.",
        "{subj} so {adj} and cheerful today.",
        "{subj} {intensifier} {adj} and delighted.",
        "We are so {adj} and grateful for everything.",
        "I am in such a good mood, I feel {adj} and peaceful.",
        "I feel full of energy, happiness, and {adj}."
    ],
    "Panic": [
        "I can't breathe, I feel {intensifier} {adj}!",
        "My chest is so tight, I am having a {adj} attack.",
        "Everything is spinning and I feel {intensifier} {adj}.",
        "{subj} feeling like I'm losing control and completely {adj}.",
        "I need to get out of here right now, I'm so {adj}!",
        "I am absolutely terrified and feel {adj}.",
        "I feel a sudden surge of fear, I am {intensifier} {adj}.",
        "I feel suffocated, trapped, and {adj}.",
        "My heart is pounding and I'm feeling so {adj} and scared.",
        "I'm shaking uncontrollably, I feel {intensifier} {adj}.",
        "{subj} {intensifier} {adj} and hyperventilating.",
        "{subj} {adj} and trapped in this room.",
        "{subj} having a severe {adj} attack right now.",
        "I'm losing my mind and feeling so {adj}.",
        "I am terrified, sweating, and feeling {intensifier} {adj}."
    ],
    "Sadness": [
        "{subj} so {adj} and like crying all the time.",
        "I miss them so much, I am {intensifier} {adj}.",
        "My heart is broken, I feel {intensifier} {adj}.",
        "{subj} feeling really down and {adj} today.",
        "I can't stop crying, I'm so {adj}.",
        "It hurts so much inside, I feel {intensifier} {adj}.",
        "I feel incredibly lonely, blue, and {adj}.",
        "Everything feels so grey and I'm feeling {adj}.",
        "I feel a deep sense of sorrow and {adj}.",
        "I am dejected and feeling extremely {adj}.",
        "{subj} {adj} and is grieving.",
        "{subj} {intensifier} {adj} and tearful.",
        "They feel incredibly {adj} and lonely.",
        "My heart is heavy and I feel so {adj}.",
        "I'm having a really sad, tearful, and {adj} day."
    ],
    "Stress": [
        "I have way too much work, I feel {intensifier} {adj}.",
        "These deadlines are killing me, I am so {adj}.",
        "{subj} completely overwhelmed and {adj} with tasks.",
        "{subj} feeling {intensifier} {adj} and close to burning out.",
        "I have no time to rest, I feel {intensifier} {adj}.",
        "There is so much pressure on me, I am {adj}.",
        "My head is about to explode from feeling so {adj}.",
        "{subj} completely exhausted, tired, and {adj}.",
        "I feel drained and {adj} by my daily responsibilities.",
        "I can't cope with this pressure, I feel so {adj}.",
        "{subj} {intensifier} {adj} and overworked.",
        "{subj} {intensifier} {adj} and stretched to the limit.",
        "They are highly {adj} due to their work load.",
        "I am feeling incredibly {adj} and weary today.",
        "I feel so tired, strained, and {adj} by life."
    ]
}

# =====================================================================
# STEP 2: PROGRAMMATIC CORPUS GENERATION
# =====================================================================
print("Generating corpus...")
X_raw = []
y_raw = []

# Generate 2000 unique records per category to have a highly balanced 14,000 sentence dataset
for category in synonyms.keys():
    cat_syns = synonyms[category]
    cat_templates = templates[category]
    cat_dataset = set()
    
    # Try generating variations
    attempts = 0
    while len(cat_dataset) < 2050 and attempts < 200000:
        attempts += 1
        template = random.choice(cat_templates)
        subj = random.choice(subjects)
        intensifier = random.choice(intensifiers)
        adj = random.choice(cat_syns)
        ctx = random.choice(contexts)
        
        sentence = template.format(
            subj=subj,
            intensifier=intensifier,
            adj=adj,
            context=ctx
        )
        cat_dataset.add(sentence)
        
    print(f"Generated {len(cat_dataset)} unique samples for class: {category}")
    for sentence in cat_dataset:
        X_raw.append(sentence)
        y_raw.append(category)

# Shuffle the dataset
combined = list(zip(X_raw, y_raw))
random.shuffle(combined)
X_raw, y_raw = zip(*combined)

X_raw = list(X_raw)
y_raw = list(y_raw)

# =====================================================================
# STEP 3: TEXT PREPROCESSING & CLEANING
# =====================================================================
print("Preprocessing and cleaning text...")
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

X_clean = [clean_text(text) for text in X_raw]

# =====================================================================
# STEP 4: FIT TOKENIZER & LABEL ENCODER
# =====================================================================
print("Fitting Tokenizer and LabelEncoder...")
vocab_size = 5000
max_length = 50

# Using oov_token and standard Keras Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_clean)

# Fit LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

# Save Pickles immediately to current directory
print("Saving tokenizer.pkl and label_encoder.pkl...")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print(f"Tokenizer vocab size: {len(tokenizer.word_index)}")
print(f"Label Encoder classes: {label_encoder.classes_}")

# =====================================================================
# STEP 5: PREPARE SEQUENCES
# =====================================================================
sequences = tokenizer.texts_to_sequences(X_clean)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Split train and validation (85% train, 15% validation)
split_idx = int(0.85 * len(padded_sequences))
X_train, X_val = padded_sequences[:split_idx], padded_sequences[split_idx:]
y_train, y_val = y_encoded[:split_idx], y_encoded[split_idx:]

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# =====================================================================
# STEP 6: CONSTRUCT THE SIMPLERNN MODEL
# =====================================================================
print("Building the Keras SimpleRNN Model...")

model = Sequential([
    # Embedding: input_dim=5000, output_dim=64, input_length=50
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    
    # SimpleRNN with 64 units, returning sequences to let pooling capture structural peak
    SimpleRNN(64, return_sequences=True, activation='tanh'),
    
    # MaxPooling to extract the most prominent activation for classification
    GlobalMaxPooling1D(),
    
    # Fully connected layers with regularization
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    # Softmax output over 7 categories
    Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =====================================================================
# STEP 7: TRAIN THE MODEL
# =====================================================================
print("Training the RNN model...")
history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# =====================================================================
# STEP 8: SAVE THE MODEL & VERIFY
# =====================================================================
print("Saving the model to upgraded_RNN_model.h5...")
model.save("upgraded_RNN_model.h5")
print("Model successfully saved!")

print("\n--- Pipeline Completed Successfully! ---")
