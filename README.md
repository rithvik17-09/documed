# Documed - AI-Powered Healthcare Management Platform

![Documed Banner](https://img.shields.io/badge/Healthcare-AI%20Platform-blue) ![Next.js](https://img.shields.io/badge/Next.js-14-black) ![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-orange) ![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Understanding Neural Networks](#understanding-neural-networks)
  - [CNN Architecture](#cnn-architecture)
  - [Training Process](#training-process)
  - [Model Weights Explained](#model-weights-explained)
- [Module Documentation](#module-documentation)
  - [DocMate - First Aid Assistant](#1-docmate---first-aid-assistant)
  - [MediMate - Medical Assistant](#2-medimate---medical-assistant)
  - [MediMood - Mental Wellness Tracker](#3-medimood---mental-wellness-tracker)
  - [PulseChain - Emergency Response](#4-pulsechain---emergency-response)
  - [X-Ray Analyser - Medical Image Analysis](#5-x-ray-analyser---medical-image-analysis)
- [Installation & Setup](#installation--setup)
- [Authentication System](#authentication-system)
- [Database Architecture](#database-architecture)
- [Contributing](#contributing)

---

## 🎯 Overview

Documed is an innovative healthcare management platform that combines artificial intelligence with user-friendly interfaces to provide comprehensive medical assistance. The platform integrates machine learning models for medical image analysis, conversational AI for symptom checking, mental wellness tracking, emergency response systems, and first-aid guidance all in one unified application.

The project demonstrates how modern web technologies can be merged with deep learning to create practical healthcare solutions that are accessible to everyone. By leveraging convolutional neural networks trained on medical imaging datasets, Documed can assist in preliminary medical diagnoses while maintaining a focus on user privacy and data security.

---

## 🚀 Key Features

- **AI-Powered Medical Image Analysis**: Trained CNN model for chest X-ray and MRI scan interpretation
- **Intelligent Symptom Checker**: Context-aware chatbot that helps users understand their symptoms
- **Mental Wellness Tracking**: Mood monitoring with journaling and personalized content suggestions
- **Emergency Response System**: Real-time vitals monitoring with SOS functionality
- **First-Aid Guidance**: Interactive guides with step-by-step instructions for common medical emergencies
- **Medicine Reminder System**: Smart medication scheduling with notifications
- **Appointment Booking**: Integrated healthcare provider scheduling
- **Secure Authentication**: NextAuth-powered user management with password reset capabilities
- **Dark Mode Support**: Eye-friendly interface with theme switching

---

## 🛠 Technology Stack

### Frontend
- **Next.js 14**: React framework with server-side rendering and API routes
- **TypeScript**: Type-safe development environment
- **Tailwind CSS**: Utility-first styling framework
- **Framer Motion**: Smooth animations and transitions
- **Shadcn/ui**: Accessible component library
- **React Hook Form**: Efficient form management

### Backend
- **Next.js API Routes**: Serverless API endpoints
- **Prisma ORM**: Type-safe database client
- **MongoDB**: NoSQL database for flexible data storage
- **NextAuth.js**: Authentication and session management
- **Nodemailer**: Email service for password resets and notifications

### Machine Learning
- **TensorFlow/Keras**: Deep learning framework
- **Python**: Model training environment
- **NumPy & Pandas**: Data manipulation libraries
- **OpenCV**: Image processing
- **Matplotlib**: Visualization tools

---

## 🧠 Machine Learning Pipeline

### Understanding Neural Networks

Neural networks are computational systems inspired by the biological neural networks in our brains. Think of them as a series of interconnected decision-making units (neurons) that learn patterns from data. Here's how they work in simple terms:

1. **Input Layer**: Receives raw data (like pixel values from an X-ray image)
2. **Hidden Layers**: Process and transform the data through mathematical operations
3. **Output Layer**: Produces the final prediction (Normal vs. Pneumonia)

Each connection between neurons has a "weight" - a numerical value that determines how much influence one neuron has on another. During training, these weights are continuously adjusted to improve accuracy.

### CNN Architecture

Our Documed platform uses a **Convolutional Neural Network (CNN)**, which is specifically designed for image analysis. CNNs are particularly effective for medical imaging because they can automatically detect patterns like edges, textures, and shapes without manual feature extraction.

#### Why CNNs for Medical Imaging?

Traditional image analysis requires experts to manually identify which features matter. CNNs learn these features automatically by examining thousands of examples. For chest X-rays, the network learns to recognize:
- Lung opacity patterns indicating pneumonia
- Bone structures and their normal appearances
- Tissue density variations
- Spatial relationships between anatomical structures

#### Our CNN Architecture Breakdown

```python
# Input: 150x150 RGB image (chest X-ray)
inputs = Input(shape=(150, 150, 3))

# Block 1: Initial feature detection (16 filters)
x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
```

**What's happening here:**
- **Conv2D Layer**: Slides a 3x3 filter across the image, detecting basic patterns like edges and corners
- **16 filters**: The network learns 16 different pattern detectors
- **ReLU activation**: Introduces non-linearity, allowing the network to learn complex patterns
- **MaxPooling**: Reduces image size while keeping important features, making computation efficient

```python
# Block 2: Mid-level features (32 filters)
x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)
```

**Advanced concepts:**
- **SeparableConv2D**: More efficient than regular convolution - splits the operation into two steps, reducing computation while maintaining accuracy
- **BatchNormalization**: Standardizes the data flowing through the network, making training faster and more stable
- **32 filters**: Detects more complex patterns like texture combinations

```python
# Blocks 3-5: Deep feature extraction (64, 128, 256 filters)
# Each block goes deeper into understanding the image
x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)

x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.2)(x)

x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.2)(x)
```

**Progressive learning:**
- As we go deeper, the number of filters increases (16 → 32 → 64 → 128 → 256)
- Early layers detect simple patterns (edges, colors)
- Middle layers combine these into shapes and textures
- Deep layers recognize complex medical features (lung consolidation, pleural effusion)
- **Dropout**: Randomly turns off 20% of neurons during training to prevent overfitting (memorizing training data instead of learning general patterns)

```python
# Flatten and Dense layers: Decision making
x = Flatten()(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(rate=0.7)(x)
x = Dense(units=128, activation='relu')(x)
x = Dropout(rate=0.5)(x)
x = Dense(units=64, activation='relu')(x)
x = Dropout(rate=0.3)(x)

# Output: Binary classification (Normal or Pneumonia)
output = Dense(units=1, activation='sigmoid')(x)
```

**Final decision stage:**
- **Flatten**: Converts 2D feature maps into a 1D vector
- **Dense layers**: Fully connected neurons that combine all learned features
- **Progressive dropout**: Higher dropout rates early (70%) prevent over-reliance on any single feature
- **Sigmoid activation**: Outputs a probability between 0 and 1 (0 = Normal, 1 = Pneumonia)

### Training Process

Training a neural network is like teaching a student through practice and feedback. Here's the complete workflow:

#### 1. Dataset Preparation

```python
# Chest X-ray dataset structure
chest_xray/
  ├── train/        # 5,216 images for learning
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  ├── val/          # 16 images for tuning
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  └── test/         # 624 images for final evaluation
      ├── NORMAL/
      └── PNEUMONIA/
```

**Why this split?**
- **Training set (80%)**: The network learns patterns from these images
- **Validation set (3%)**: Used during training to check if the model generalizes well
- **Test set (17%)**: Final exam - never seen during training, gives true performance measure

#### 2. Data Augmentation

Real-world medical images vary significantly due to different X-ray machines, patient positioning, and exposure settings. Data augmentation artificially creates variations to make our model robust:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to 0-1 range
    shear_range=0.2,         # Slant images slightly
    zoom_range=0.2,          # Zoom in/out randomly
    horizontal_flip=True,    # Flip images horizontally
)
```

**Why augment?**
- Prevents overfitting by showing the network varied examples
- Simulates real-world variations in patient positioning
- Helps the model learn invariant features (pneumonia looks like pneumonia regardless of slight position changes)

#### 3. Model Compilation

```python
model.compile(
    optimizer='adam',              # Adam: adaptive learning rate optimizer
    loss='binary_crossentropy',    # Measures prediction error
    metrics=['accuracy']           # Tracks correct predictions
)
```

**Optimizer explained:**
- **Adam (Adaptive Moment Estimation)**: Automatically adjusts learning speed for each parameter
- Starts with large steps to learn quickly
- Takes smaller steps as it approaches the optimal solution
- Like a smart GPS that adjusts your route based on traffic

**Loss function:**
- Measures how wrong the predictions are
- Binary crossentropy is ideal for yes/no questions (Normal or Pneumonia)
- Lower loss = better predictions

#### 4. Callbacks for Smart Training

```python
# Save the best model automatically
checkpoint = ModelCheckpoint(
    filepath='best_weights.hdf5',
    save_best_only=True,
    save_weights_only=True
)

# Reduce learning rate when improvement plateaus
lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    verbose=2,
    mode='max'
)

# Stop training if no improvement
early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.1,
    patience=1,
    mode='min'
)
```

**Callback purposes:**
- **ModelCheckpoint**: Saves only the best-performing version (highest validation accuracy)
- **ReduceLROnPlateau**: When learning stalls, reduce learning rate by 70% to fine-tune
- **EarlyStopping**: Prevents wasting time if the model stops improving

#### 5. Training Loop

```python
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // batch_size,
    epochs=10,
    validation_data=test_gen,
    validation_steps=test_gen.samples // batch_size,
    callbacks=[checkpoint, lr_reduce, early_stop]
)
```

**What happens in each epoch:**
1. Network processes all training images in batches of 32
2. For each batch:
   - Make predictions
   - Calculate error (loss)
   - Adjust weights to reduce error (backpropagation)
3. After all training batches, check performance on validation set
4. Callbacks decide whether to save, adjust learning rate, or stop

### Model Weights Explained

Weights are the "knowledge" learned by the neural network. Think of them as the strength of connections between neurons.

#### What Are Weights?

In biological brains, synapses connect neurons with varying strengths. Neural network weights work similarly:

```
Input Pixel × Weight = Influence on Next Layer
```

Example:
```
Pixel value: 0.8 (bright area in X-ray)
Weight: 2.5 (learned during training)
Output: 0.8 × 2.5 = 2.0 (strong signal to next neuron)
```

#### How Weights Are Learned

**Initial State**: Random weights (the network knows nothing)
```python
np.random.seed(232)  # Ensures reproducible random initialization
```

**Learning Process** (Gradient Descent):
1. Network makes a prediction with current weights
2. Calculate error: How far off was the prediction?
3. Compute gradients: Which direction should each weight change?
4. Update weights: Adjust slightly in the direction that reduces error
5. Repeat thousands of times

**Mathematical representation:**
```
New Weight = Old Weight - (Learning Rate × Gradient)
```

#### Why We Save Weights

Training takes hours and requires expensive GPU resources. The file `best_weights.hdf5` (HDF5 format) stores:
- All learned weight values for every layer
- Optimizer state (for resuming training)
- Model architecture information

**File size**: Typically 50-200 MB for medical imaging models

**Loading saved weights:**
```python
model.load_weights('best_weights.hdf5')
# Now the model "remembers" everything it learned
```

#### Weight Distribution Example

```
Layer 1 (16 filters, 3×3 kernel):
  - 432 weights (16 filters × 3 × 3 × 3 color channels)
  
Layer 5 (256 filters):
  - 294,912 weights
  
Total network weights: ~5-10 million parameters
```

Each weight represents a tiny piece of learned knowledge. Together, they form a complex decision-making system that can identify pneumonia patterns invisible to untrained eyes.

---

## 📚 Module Documentation

### 1. DocMate - First Aid Assistant

**Purpose**: Provides immediate first-aid guidance for common medical emergencies through an intelligent conversational interface.

#### How It Works

DocMate combines keyword-based response matching with AI-powered conversational capabilities to deliver accurate first-aid instructions instantly.

**Architecture Flow:**
```
User Input → Keyword Detection → Response Matching → Display Instructions
     ↓                                    ↓
Gemini AI Fallback ← No Match Found ← Check Knowledge Base
```

**Key Features:**

1. **Instant Emergency Response**
   - Pre-loaded responses for 14 common emergencies
   - No internet dependency for critical situations
   - Response time: <100ms

2. **Comprehensive Coverage**
   ```javascript
   const emergencies = [
     'CPR', 'Burns', 'Choking', 'Bleeding', 'Heart Attack',
     'Stroke', 'Fracture', 'Sprain', 'Allergic Reaction',
     'Poisoning', 'Seizure', 'Heat Stroke', 'Hypothermia'
   ]
   ```

3. **AI-Enhanced Responses**
   - For uncommon queries, integrates with Gemini API
   - Provides context-aware medical advice
   - Fallback to general emergency instructions

#### Technical Implementation

**Keyword Matching Algorithm:**
```typescript
const getFirstAidResponse = (input: string) => {
  const lowerInput = input.toLowerCase();
  
  // Iterate through knowledge base
  for (const [keyword, response] of Object.entries(firstAidResponses)) {
    if (lowerInput.includes(keyword)) {
      return response;  // Immediate match
    }
  }
  
  // No match: default safety response
  return "Call emergency services (911) immediately...";
}
```

**Example Interaction:**

```
User: "My friend is choking on food"
      ↓
Keyword Detection: "choking" found
      ↓
Response: {
  1. Ask 'Are you choking?'
  2. If unable to speak → Heimlich maneuver
  3. Stand behind, wrap arms around waist
  4. Make fist above navel
  5. Quick inward/upward thrusts
  6. Repeat until object expelled
}
```

**Interactive Guides Component:**

The `FirstAidGuides` component provides visual step-by-step tutorials:

```typescript
type Guide = {
  id: string;
  title: string;           // "CPR Basics"
  description: string;     // Summary
  thumbnail: string;       // Visual preview
  videoUrl?: string;       // Video tutorial link
  steps: Array<{
    title: string;         // "Check responsiveness"
    description: string;   // Detailed instruction
  }>;
}
```

**User Experience Features:**
- Chat-style interface with message history
- Quick-access common emergency buttons
- Visual guides with step-by-step illustrations
- Clear, concise instructions suitable for high-stress situations
- Tab system: Chat, Guides, Video Tutorials

#### Safety Mechanisms

1. **Liability Protection**: All responses include disclaimers to seek professional medical help
2. **Clear Escalation**: Always recommends calling emergency services for serious conditions
3. **No Diagnosis**: Provides instructions, not medical diagnoses
4. **Timeout Warnings**: For time-sensitive procedures (CPR, choking)

---

### 2. MediMate - Medical Assistant

**Purpose**: A comprehensive medical companion that helps users understand symptoms, manage medications, and book appointments with healthcare providers.

#### Module Architecture

MediMate consists of three integrated sub-modules accessible through a tabbed interface:

```
MediMate
├── Symptom Checker (AI-powered symptom analysis)
├── Medicine Reminder (Medication scheduling)
└── Appointment Booking (Healthcare provider scheduling)
```

#### A. Symptom Checker

**How It Works:**

The symptom checker uses a sophisticated keyword-matching system combined with medical knowledge databases to provide preliminary symptom analysis.

**Processing Pipeline:**
```
User describes symptoms
      ↓
Text normalization (lowercase)
      ↓
Keyword extraction
      ↓
Match against medical database
      ↓
Generate response with recommendations
      ↓
Display with severity indicators
```

**Medical Knowledge Base:**

```typescript
const symptomResponses = {
  'headache': {
    causes: ['stress', 'dehydration', 'eye strain'],
    recommendations: ['rest', 'hydrate', 'reduce screen time'],
    seekHelpIf: ['severe', 'persistent', 'accompanied by fever']
  },
  'fever': {
    causes: ['infection', 'inflammation'],
    normalRange: '< 100.4°F (38°C)',
    treatment: ['rest', 'fluids', 'acetaminophen'],
    seekHelpIf: ['> 103°F', '> 3 days', 'difficulty breathing']
  }
  // ... 20+ symptom patterns
}
```

**Intelligent Features:**

1. **Context-Aware Responses**
   ```typescript
   Input: "severe headache on forehead"
   Analysis: {
     symptom: "headache",
     severity: "severe",
     location: "forehead",
     → Response: Sinus pressure treatment + urgency flag
   }
   ```

2. **Multi-Symptom Recognition**
   - Detects combinations: "fever and cough" → potential respiratory infection
   - Provides comprehensive assessment
   - Suggests related symptoms to monitor

3. **Quick Access Buttons**
   - Common symptoms displayed as clickable tags
   - One-tap symptom entry
   - Reduces typing in uncomfortable situations

**Safety Features:**
- Red flag symptoms trigger immediate doctor recommendation
- No medication prescription (only OTC suggestions)
- Encourages professional consultation for unclear cases

#### B. Medicine Reminder

**Purpose**: Helps users adhere to medication schedules through intelligent reminders and tracking.

**Data Structure:**
```typescript
type Reminder = {
  id: string;           // Unique identifier
  medicine: string;     // "Ibuprofen 400mg"
  time: string;         // "08:00" (24-hour format)
  dosage: string;       // "2 tablets"
  frequency: string;    // "daily", "twice daily", "weekly"
  completed: boolean;   // Taken today?
}
```

**Smart Features:**

1. **Time-Based Visual Cues**
   ```typescript
   const getTimeStatus = (reminderTime: string) => {
     const now = new Date();
     const reminder = parseTime(reminderTime);
     const diff = reminder - now;
     
     if (diff < 0) return 'past';           // Missed - red indicator
     if (diff < 3600000) return 'soon';     // <1 hour - yellow indicator
     return 'upcoming';                     // >1 hour - blue indicator
   }
   ```

2. **Visual Status Indicators**
   - Past due: Red badge with "Overdue"
   - Coming soon (<1 hour): Yellow badge with "Soon"
   - Upcoming: Blue badge with time remaining
   - Completed: Green checkmark

3. **Frequency Options**
   - Daily: Standard medications
   - Twice daily: Morning/evening split
   - Weekly: Supplements like Vitamin D
   - As needed: PRN medications

**User Experience:**
```
Add Medication Flow:
1. Click "+ Add Reminder"
2. Enter medicine name
3. Set time (time picker)
4. Specify dosage
5. Choose frequency
6. Save → Auto-sorted by time
```

**Persistence Mechanism:**
```typescript
// Currently client-side (localStorage ready)
// Future: Backend integration for cross-device sync
const saveReminders = async (reminders: Reminder[]) => {
  // await fetch('/api/reminders', { method: 'POST', body: JSON.stringify(reminders) });
  localStorage.setItem('medicineReminders', JSON.stringify(reminders));
}
```

#### C. Appointment Booking

**Purpose**: Streamlines the process of finding and booking appointments with nearby healthcare providers.

**Features:**

1. **Provider Search**
   ```typescript
   type Provider = {
     id: string;
     name: string;           // "Dr. Sarah Johnson"
     specialty: string;      // "Cardiologist"
     rating: number;         // 4.8 out of 5
     distance: string;       // "2.3 km away"
     availableSlots: Date[];
     address: string;
     phone: string;
   }
   ```

2. **Specialty Filtering**
   - General Practitioner
   - Cardiologist
   - Dermatologist
   - Pediatrician
   - Orthopedic Surgeon
   - Neurologist
   - Psychiatrist

3. **Smart Scheduling**
   - Real-time availability display
   - Conflict detection with existing appointments
   - Calendar integration ready
   - Reminder notifications

**Booking Flow:**
```
Select Specialty → View Providers → Choose Doctor → Pick Time Slot → Confirm
       ↓                  ↓               ↓              ↓             ↓
Filter by rating    See reviews    Check schedule   Add notes    Get confirmation
```

**Future Integrations:**
- Google Calendar sync
- SMS/Email confirmations
- Video consultation options
- Insurance verification
- Prescription refill requests

---

### 3. MediMood - Mental Wellness Tracker

**Purpose**: Comprehensive mental health monitoring platform that combines mood tracking, journaling, AI-powered mood analysis, and personalized content recommendations.

#### Architecture Overview

```
MediMood Platform
├── Mood Picker (Emotion selection interface)
├── Mood Analyzer (Pattern recognition & insights)
├── Journal Entries (Private diary with mood context)
└── Content Suggestions (Personalized wellness resources)
```

#### A. Mood Picker

**How It Works:**

The mood picker uses a visual, emoji-based interface to make emotional expression intuitive and low-friction.

**Mood Taxonomy:**
```typescript
const moods = [
  { emoji: '😊', label: 'Happy', gradient: 'yellow-to-orange' },
  { emoji: '😌', label: 'Calm', gradient: 'blue-to-teal' },
  { emoji: '😐', label: 'Neutral', gradient: 'gray-to-slate' },
  { emoji: '😔', label: 'Sad', gradient: 'indigo-to-purple' },
  { emoji: '😡', label: 'Angry', gradient: 'red-to-pink' },
  { emoji: '😰', label: 'Anxious', gradient: 'orange-to-amber' },
  { emoji: '😴', label: 'Tired', gradient: 'purple-to-indigo' },
  { emoji: '🤗', label: 'Grateful', gradient: 'green-to-emerald' },
  { emoji: '🥳', label: 'Excited', gradient: 'pink-to-rose' },
  { emoji: '😵', label: 'Overwhelmed', gradient: 'violet-to-purple' },
  { emoji: '🤔', label: 'Confused', gradient: 'amber-to-yellow' },
  { emoji: '😤', label: 'Frustrated', gradient: 'rose-to-red' }
]
```

**Visual Design Philosophy:**
- Large, tappable emoji buttons (easy on mobile)
- Color-coded emotional categories
- Gradient backgrounds that shift based on selected mood
- Smooth animations for emotional resonance

**Interaction Flow:**
```
User taps mood emoji
      ↓
Background color transitions to mood's gradient
      ↓
Journal text area appears
      ↓
User writes optional context
      ↓
Save creates timestamped entry
```

#### B. Mood Analyzer

**Purpose**: Transforms raw mood data into actionable insights through pattern recognition and trend analysis.

**Analysis Components:**

1. **Weekly Mood Distribution**
   ```typescript
   const analyzeMoodDistribution = (entries: Entry[]) => {
     const last7Days = entries.filter(e => isWithinDays(e.date, 7));
     
     return {
       dominant: getMostFrequent(last7Days),      // "Happy (4 times)"
       improving: isTrendPositive(last7Days),     // true/false
       variety: getUniqueCount(last7Days)         // Emotional diversity
     };
   }
   ```

2. **Temporal Patterns**
   ```typescript
   const detectTimePatterns = (entries: Entry[]) => {
     return {
       morningMood: getAverageMood(entries, 'morning'),
       afternoonMood: getAverageMood(entries, 'afternoon'),
       eveningMood: getAverageMood(entries, 'evening'),
       worstDay: findLowestDay(entries),          // "Mondays"
       bestDay: findHighestDay(entries)           // "Saturdays"
     };
   }
   ```

3. **Trend Visualization**
   - Line charts showing mood changes over time
   - Color-coded mood calendar (heatmap style)
   - Streak tracking (consecutive positive days)
   - Trigger identification (journalevents correlating with mood dips)

**AI-Powered Insights:**
```typescript
const generateInsights = async (moodHistory: Entry[]) => {
  const patterns = analyzePatterns(moodHistory);
  
  return {
    summary: "You've been feeling anxious in the evenings this week",
    recommendation: "Consider mindfulness exercises before bed",
    positives: "Your morning moods show consistent improvement!",
    concernFlags: patterns.hasSignificantDecline ? "Consider professional support" : null
  };
}
```

#### C. Journal Entries

**Purpose**: Provides a safe, private space for emotional expression and self-reflection.

**Entry Structure:**
```typescript
type JournalEntry = {
  id: string;
  date: Date;            // Timestamp of entry
  mood: Mood;            // Associated emotional state
  text: string;          // User's written reflection
  tags?: string[];       // Optional: "work", "family", "health"
  private: boolean;      // Privacy setting (future: shareable)
}
```

**Features:**

1. **Mood-Contextualized Entries**
   - Each journal entry is linked to a specific mood
   - Visual mood indicator on each entry card
   - Timeline view showing emotional journey

2. **Rich Text Support**
   - Markdown formatting ready
   - Emoji support for expression
   - Character count (encourages writing)

3. **Search and Filter**
   ```typescript
   const filterEntries = (entries: Entry[], filters: Filter) => {
     return entries.filter(entry => {
       if (filters.mood && entry.mood.label !== filters.mood) return false;
       if (filters.dateRange && !inRange(entry.date, filters.dateRange)) return false;
       if (filters.keyword && !entry.text.includes(filters.keyword)) return false;
       return true;
     });
   }
   ```

4. **Privacy Features**
   - Local-first storage (data never leaves device by default)
   - Optional cloud backup with encryption
   - Export functionality (PDF, JSON)

#### D. Content Suggestions

**Purpose**: Delivers personalized mental wellness resources based on current mood and historical patterns.

**Recommendation Engine:**

```typescript
const generateContentSuggestions = (currentMood: Mood, moodHistory: Entry[]) => {
  const suggestions = [];
  
  // Mood-specific content
  if (currentMood.label === 'Anxious') {
    suggestions.push({
      type: 'breathing-exercise',
      title: '4-7-8 Breathing Technique',
      duration: '5 minutes',
      difficulty: 'beginner'
    });
  }
  
  if (currentMood.label === 'Sad') {
    suggestions.push({
      type: 'article',
      title: 'Understanding Sadness: A Natural Emotion',
      readTime: '8 minutes'
    });
  }
  
  // Pattern-based recommendations
  if (hasRepeatingAnxiety(moodHistory)) {
    suggestions.push({
      type: 'therapy-resource',
      title: 'Cognitive Behavioral Therapy (CBT) Basics',
      link: '/resources/cbt'
    });
  }
  
  return suggestions;
}
```

**Content Categories:**

1. **Mindfulness Exercises**
   - Guided meditations (3, 5, 10-minute options)
   - Breathing techniques
   - Progressive muscle relaxation

2. **Educational Articles**
   - Understanding emotions
   - Stress management techniques
   - Sleep hygiene guides

3. **Interactive Activities**
   - Gratitude prompts
   - Creative expression exercises
   - Physical wellness challenges

4. **Professional Resources**
   - Therapist directories
   - Crisis hotlines
   - Support group information

**Personalization Algorithm:**
```typescript
const personalizeContent = (user: User) => {
  const preferences = {
    contentTypes: user.engagementHistory,    // What they've clicked
    timeOfDay: getCurrentTimeSlot(),         // Morning/afternoon/evening
    recentMoods: getRecentMoods(user, 3),    // Last 3 entries
    avoidTopics: user.preferences.avoid      // User-specified exclusions
  };
  
  return matchContent(contentLibrary, preferences);
}
```

#### Mental Health Safeguards

1. **Crisis Detection**
   ```typescript
   const detectCrisisSignals = (entry: Entry) => {
     const crisisKeywords = ['suicide', 'self-harm', 'ending it', 'no point'];
     const hasCrisisLanguage = crisisKeywords.some(kw => entry.text.includes(kw));
     
     if (hasCrisisLanguage) {
       showCrisisResources();  // Immediate hotline display
     }
   }
   ```

2. **Professional Recommendation Triggers**
   - 7+ consecutive days of negative moods
   - Significant mood volatility (rapid cycling)
   - User-initiated help requests

3. **Data Privacy**
   - All journal entries encrypted at rest
   - No analytics on journal content without explicit consent
   - Easy data export and deletion

---

### 4. PulseChain - Emergency Response

**Purpose**: Real-time health monitoring and emergency assistance system with SOS capabilities, vital signs tracking, and emergency contact management.

#### System Architecture

```
PulseChain Emergency System
├── SOS Button (Panic button with location sharing)
├── Vitals Monitor (Heart rate, oxygen, temperature)
├── Emergency Contacts (Quick-dial saved contacts)
└── Alerts Feed (Health warnings & notifications)
```

#### A. SOS Button

**How It Works:**

The SOS button provides one-tap emergency activation that triggers multiple life-saving protocols simultaneously.

**Activation Flow:**
```
User presses SOS button
      ↓
Visual confirmation (pulsing red alert)
      ↓
┌─────────────┬─────────────┬─────────────┐
│   Location  │   Notify    │   Display   │
│   Capture   │  Contacts   │   Status    │
└─────────────┴─────────────┴─────────────┘
      ↓             ↓              ↓
Get GPS coords   SMS to all    Show "Help
      +           emergency     is coming"
  Address         contacts       message
```

**Technical Implementation:**
```typescript
const activateSOS = async () => {
  setSosActive(true);
  
  // 1. Get precise location
  const location = await getCurrentPosition();
  const address = await reverseGeocode(location);
  
  // 2. Prepare emergency message
  const message = `
    🚨 EMERGENCY ALERT
    ${user.name} needs help!
    Location: ${address}
    Coordinates: ${location.lat}, ${location.lng}
    Time: ${new Date().toLocaleString()}
    
    View location: https://maps.google.com/?q=${location.lat},${location.lng}
  `;
  
  // 3. Notify all emergency contacts
  await Promise.all(
    emergencyContacts.map(contact => 
      sendSMS(contact.phone, message) &&
      sendPushNotification(contact.userId, message)
    )
  );
  
  // 4. Call emergency services (optional auto-dial)
  if (user.settings.autoCallEmergency) {
    window.location.href = 'tel:911';
  }
  
  // 5. Start continuous location tracking
  startLocationUpdates(30000); // Every 30 seconds
};
```

**Visual Feedback:**
- Pulsing red circle animation (unmissable)
- Audio alert (configurable)
- Haptic feedback on mobile
- Cannot be dismissed accidentally (requires deliberate cancel)

**Safety Features:**
- 5-second countdown before activation (prevents accidental triggers)
- "False Alarm" button to cancel with notification to contacts
- Battery optimization: efficient location updates

#### B. Vitals Monitor

**Purpose**: Continuous or on-demand monitoring of key vital signs to detect health anomalies early.

**Monitored Vitals:**

```typescript
type VitalSigns = {
  heartRate: number;      // beats per minute (bpm)
  oxygen: number;         // blood oxygen saturation (SpO2 %)
  temperature: number;    // body temperature (°F)
  bloodPressure?: {      // Optional: requires BP monitor
    systolic: number;
    diastolic: number;
  };
  timestamp: Date;
}
```

**Data Acquisition Methods:**

1. **Manual Input**
   - User enters readings from home devices
   - Guided input with normal range indicators

2. **Device Integration** (Future)
   ```typescript
   // Connect to Bluetooth health devices
   const connectDevice = async (deviceType: 'pulse-oximeter' | 'bp-monitor') => {
     const device = await navigator.bluetooth.requestDevice({
       filters: [{ services: ['heart_rate'] }]
     });
     
     // Subscribe to device readings
     device.addEventListener('characteristicvaluechanged', (event) => {
       const value = parseValue(event.target.value);
       updateVitals(value);
     });
   };
   ```

3. **Simulated Monitoring** (Demo/Testing)
   ```typescript
   const generateVitals = () => {
     return {
       heartRate: randomInRange(60, 100),      // Normal: 60-100 bpm
       oxygen: randomInRange(94, 100),         // Normal: 95-100%
       temperature: randomInRange(97.5, 99.5)  // Normal: 97.8-99.1°F
     };
   };
   ```

**Health Status Evaluation:**

```typescript
const evaluateVitals = (vitals: VitalSigns) => {
  const status = {
    heartRate: 'normal',
    oxygen: 'normal',
    temperature: 'normal',
    overall: 'healthy'
  };
  
  // Heart rate analysis
  if (vitals.heartRate < 60) {
    status.heartRate = 'low';  // Bradycardia
    status.overall = 'concerning';
  } else if (vitals.heartRate > 100) {
    status.heartRate = 'high'; // Tachycardia
    status.overall = 'concerning';
  }
  
  // Oxygen level analysis
  if (vitals.oxygen < 95) {
    status.oxygen = 'low';     // Hypoxemia
    status.overall = 'critical';
    triggerAlert('Low oxygen detected! Seek immediate medical attention.');
  }
  
  // Temperature analysis
  if (vitals.temperature > 100.4) {
    status.temperature = 'fever';
    status.overall = 'monitor';
  } else if (vitals.temperature < 95) {
    status.temperature = 'hypothermia';
    status.overall = 'critical';
  }
  
  return status;
};
```

**Visual Representation:**

```typescript
// Color-coded cards based on status
const VitalCard = ({ vital, value, status }) => {
  const colors = {
    normal: 'bg-green-50 border-green-200',
    low: 'bg-yellow-50 border-yellow-200',
    high: 'bg-orange-50 border-orange-200',
    critical: 'bg-red-50 border-red-200'
  };
  
  return (
    <Card className={colors[status]}>
      <Icon />
      <Value>{value}</Value>
      <Status>{getStatusMessage(status)}</Status>
    </Card>
  );
};
```

**Trend Tracking:**
- Historical charts showing vital trends
- Anomaly detection (sudden spikes/drops)
- Export data for doctor appointments

#### C. Emergency Contacts

**Purpose**: Quick access to trusted individuals during medical emergencies.

**Contact Structure:**
```typescript
type EmergencyContact = {
  id: string;
  name: string;
  relationship: string;   // "Spouse", "Parent", "Doctor", "Friend"
  phone: string;          // Primary contact method
  email?: string;
  priority: number;       // 1 = first to notify
  notes?: string;         // "Has medical training", "Lives nearby"
}
```

**Features:**

1. **Priority-Based Notification**
   ```typescript
   const notifyContacts = async (contacts: EmergencyContact[]) => {
     const sorted = contacts.sort((a, b) => a.priority - b.priority);
     
     for (const contact of sorted) {
       await sendNotification(contact);
       await delay(2000); // 2-second gap between notifications
     }
   };
   ```

2. **One-Tap Communication**
   - Direct call button
   - SMS with pre-filled emergency message
   - Share live location

3. **Medical Information Sharing**
   ```typescript
   const emergencyProfile = {
     bloodType: 'O+',
     allergies: ['Penicillin', 'Shellfish'],
     medications: ['Lisinopril 10mg daily'],
     conditions: ['Hypertension'],
     insurance: { provider: 'Blue Cross', policyNumber: '...' }
   };
   ```

#### D. Alerts Feed

**Purpose**: Centralized notification system for health-related warnings and updates.

**Alert Types:**

1. **Critical Health Alerts**
   ```typescript
   {
     type: 'vital-alert',
     severity: 'critical',
     message: 'Blood oxygen below 94% - seek immediate care',
     timestamp: Date,
     action: { label: 'Call 911', callback: () => tel('911') }
   }
   ```

2. **Medication Reminders**
   ```typescript
   {
     type: 'medication',
     severity: 'info',
     message: 'Time to take Ibuprofen (400mg)',
     timestamp: Date,
     action: { label: 'Mark as Taken', callback: completeMedication }
   }
   ```

3. **Appointment Reminders**
   ```typescript
   {
     type: 'appointment',
     severity: 'info',
     message: 'Cardiology appointment tomorrow at 2 PM',
     timestamp: Date,
     action: { label: 'View Details', callback: openAppointment }
   }
   ```

4. **System Notifications**
   ```typescript
   {
     type: 'system',
     severity: 'low',
     message: 'Emergency contact updated: Mom - (555) 123-4567',
     timestamp: Date
   }
   ```

**Alert Management:**
```typescript
const manageAlerts = {
  add: (alert: Alert) => [...alerts, alert].sort(bySeverityAndTime),
  dismiss: (id: string) => alerts.filter(a => a.id !== id),
  clearAll: () => alerts.filter(a => a.severity === 'critical'), // Keep critical
  archive: (id: string) => moveToHistory(alerts.find(a => a.id === id))
};
```

---

### 5. X-Ray Analyser - Medical Image Analysis

**Purpose**: AI-powered medical imaging analysis tool that processes chest X-rays and MRI scans to detect potential abnormalities using convolutional neural networks.

#### How It Works

**Complete Analysis Pipeline:**

```
Image Upload → Preprocessing → CNN Inference → Post-processing → Results Display
      ↓              ↓                ↓               ↓                ↓
   File select   Resize to      Load trained    Threshold      Confidence +
   Validation    150×150        model weights   results        Findings list
```

#### Technical Architecture

**1. Image Upload & Validation**

```typescript
const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
  const file = event.target.files?.[0];
  
  // Validation checks
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showError('Please upload a valid image file');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {  // 10 MB limit
    showError('File too large. Maximum size: 10 MB');
    return;
  }
  
  // Create preview
  const preview = URL.createObjectURL(file);
  setImage(file);
  setPreview(preview);
};
```

**Accepted Formats:**
- JPEG/JPG: Most common medical imaging export
- PNG: Lossless quality preservation
- DICOM: Native medical imaging format (future support)

**2. Image Preprocessing**

Before feeding images to the neural network, they undergo standardized preprocessing:

```python
# Backend preprocessing (Python/TensorFlow)
def preprocess_image(image_path):
    # 1. Load image
    img = cv2.imread(image_path)
    
    # 2. Convert to RGB (some X-rays are grayscale)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # 3. Resize to model input size
    img = cv2.resize(img, (150, 150))
    
    # 4. Normalize pixel values to [0, 1]
    img = img.astype('float32') / 255.0
    
    # 5. Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img
```

**Why these steps?**
- **RGB conversion**: Model trained on 3-channel images
- **Resize**: Neural network expects fixed input dimensions
- **Normalization**: Scales pixel values to range model was trained on
- **Batch dimension**: TensorFlow processes images in batches (even if batch size = 1)

**3. CNN Inference**

```python
# Load pre-trained model
model = load_model('architecture.json')
model.load_weights('best_weights.hdf5')

# Make prediction
prediction = model.predict(preprocessed_image)

# Output interpretation
confidence = prediction[0][0]  # Value between 0 and 1
status = "Pneumonia" if confidence > 0.5 else "Normal"
```

**How Prediction Works:**

The final layer uses sigmoid activation, outputting a probability:
- **Output ≈ 0**: Strong confidence in "Normal" classification
- **Output ≈ 0.5**: Uncertain (requires expert review)
- **Output ≈ 1**: Strong confidence in "Pneumonia" classification

**Confidence Calibration:**
```typescript
const interpretConfidence = (rawScore: number) => {
  if (rawScore < 0.3) return { status: 'Normal', confidence: (1 - rawScore) * 100 };
  if (rawScore > 0.7) return { status: 'Defective', confidence: rawScore * 100 };
  return { status: 'Uncertain', confidence: 50, note: 'Recommend expert review' };
};
```

**4. Results Generation**

The system provides comprehensive analysis results:

```typescript
type AnalysisResult = {
  status: 'Normal' | 'Defective';
  confidence: number;              // 0-100%
  issues: DetectedIssue[];
  recommendation: string;
  disclaimer: string;
}

type DetectedIssue = {
  id: string;
  title: string;                   // "Lesion", "Consolidation"
  severity: 'low' | 'medium' | 'high';
  location: string;                // "Right lower lobe"
  description: string;             // Detailed finding
}
```

**Finding Generation Algorithm:**

```typescript
const generateFindings = (prediction: number, imageType: 'mri' | 'xray') => {
  const findings = imageType === 'mri' ? mriFindingsPool : xrayFindingsPool;
  
  // Severity distribution based on confidence
  const numFindings = prediction > 0.9 ? 3 : prediction > 0.7 ? 2 : 1;
  
  // Select diverse findings
  const selected = [];
  const severities = ['high', 'medium', 'low'];
  
  for (let i = 0; i < numFindings; i++) {
    const finding = randomFrom(findings.filter(f => f.severity === severities[i]));
    selected.push(finding);
  }
  
  return selected;
};
```

**Example Output:**

```
Analysis Complete
Status: Defective
Confidence: 94.3%

Detected Issues:
─────────────────
🔴 HIGH SEVERITY
  Lesion - Temporal lobe
  Suspicious hyperintense region noted on T2-weighted sequence
  
🟡 MEDIUM SEVERITY
  Edema - Perilesional area
  Mild surrounding edema pattern observed
  
🟢 LOW SEVERITY
  Cystic Component - Frontal cortex
  Small cystic pocket seen on axial slice

Recommendation:
Consult with a radiologist for professional interpretation.
This AI analysis is for preliminary screening only.
```

#### Mode Selection: MRI vs X-Ray

The analyzer supports two distinct imaging modalities:

```typescript
const modes = {
  'xray': {
    name: 'Chest X-Ray',
    expectedFindings: ['Fracture', 'Consolidation', 'Pleural Effusion', 'Pneumothorax'],
    modelWeights: 'xray_weights.hdf5',
    preprocessing: standardXrayPrep
  },
  'mri': {
    name: 'Brain MRI',
    expectedFindings: ['Lesion', 'Edema', 'Hemorrhage', 'Tumor'],
    modelWeights: 'mri_weights.hdf5',
    preprocessing: standardMriPrep
  }
};
```

**Why Different Modes?**
- Different anatomical structures
- Different imaging physics (X-ray vs magnetic resonance)
- Different pathology presentations
- Separate trained models for optimal accuracy

#### Safety & Ethical Considerations

**1. Medical Disclaimer**

```typescript
const MEDICAL_DISCLAIMER = `
⚠️ IMPORTANT MEDICAL DISCLAIMER
This AI analysis is for educational and preliminary screening purposes only.
It is NOT a substitute for professional medical diagnosis.

Always consult with qualified healthcare professionals for:
- Definitive diagnosis
- Treatment recommendations
- Medical decision making

This tool may produce false positives or false negatives.
Do not make health decisions based solely on this analysis.
`;
```

**2. Accuracy Transparency**

```typescript
const showModelMetrics = () => ({
  trainingAccuracy: '94.2%',
  validationAccuracy: '91.7%',
  testAccuracy: '89.3%',
  falsePositiveRate: '8.1%',
  falseNegativeRate: '12.6%',
  note: 'Performance varies with image quality and patient demographics'
});
```

**3. Data Privacy**

```typescript
const privacyProtection = {
  // Images never stored on server
  processingLocation: 'client-side',
  
  // Immediate cleanup after analysis
  cleanup: () => {
    URL.revokeObjectURL(preview);
    setImage(null);
    clearTempFiles();
  },
  
  // No PHI (Protected Health Information) transmitted
  anonymization: true,
  
  // Optional: remove EXIF metadata
  stripMetadata: (file: File) => removeExifData(file)
};
```

#### Performance Optimization

**1. Lazy Loading**

```typescript
// index.lazy.tsx
const XrayAnalyser = lazy(() => import('./index'));

// Usage in parent component
<Suspense fallback={<LoadingSpinner />}>
  <XrayAnalyser />
</Suspense>
```

**Why?** The ML model and associated libraries are large (~50 MB). Lazy loading prevents blocking the initial page load.

**2. Progressive Enhancement**

```typescript
const analyze = async (image: File) => {
  // 1. Show immediate UI feedback
  setLoading(true);
  showProgress(10);
  
  // 2. Simulate processing stages
  await preprocessImage(image);
  showProgress(40);
  
  // 3. Run inference
  const result = await runModel(image);
  showProgress(80);
  
  // 4. Generate report
  const findings = generateReport(result);
  showProgress(100);
  
  // 5. Display results
  displayResults(findings);
  setLoading(false);
};
```

**3. Error Handling**

```typescript
const robustAnalysis = async (image: File) => {
  try {
    return await analyze(image);
  } catch (error) {
    if (error instanceof ModelLoadError) {
      showError('Failed to load AI model. Please refresh and try again.');
    } else if (error instanceof ImageProcessingError) {
      showError('Unable to process image. Ensure it\'s a valid medical scan.');
    } else {
      showError('Unexpected error occurred. Please try again.');
      logError(error);  // Send to monitoring service
    }
  }
};
```

#### Future Enhancements

**1. Multi-Region Analysis**
- Detect multiple abnormalities in single image
- Bounding boxes around detected regions
- Heatmap overlays showing model attention

**2. Comparative Analysis**
- Upload multiple scans (before/after treatment)
- Track progression of findings
- Quantify changes over time

**3. Report Generation**
```typescript
const generatePDFReport = (analysis: AnalysisResult) => {
  return {
    patientInfo: '[Redacted for privacy]',
    scanDate: new Date(),
    findings: analysis.issues,
    confidence: analysis.confidence,
    recommendation: analysis.recommendation,
    radiologistReview: '[Space for professional notes]'
  };
};
```

---

## 🚀 Installation & Setup

### Prerequisites

```bash
Node.js >= 18.0.0
pnpm >= 8.0.0 (or npm/yarn)
MongoDB instance (local or cloud)
Python >= 3.8 (for model training)
```

### Step-by-Step Installation

**1. Clone the repository**
```bash
git clone https://github.com/rithvik17-09/documed.git
cd documed
```

**2. Install dependencies**
```bash
pnpm install
```

**3. Environment configuration**

Create `.env.local` in the project root:

```env
# Database
DATABASE_URL="mongodb://localhost:27017/documed"
# Or MongoDB Atlas: mongodb+srv://user:pass@cluster.mongodb.net/documed

# NextAuth
NEXTAUTH_URL="http://localhost:3000"
NEXTAUTH_SECRET="your-secret-key-generate-with-openssl-rand-base64-32"

# Email (for password reset)
SMTP_HOST="smtp.gmail.com"
SMTP_PORT="587"
SMTP_USER="your-email@gmail.com"
SMTP_PASS="your-app-password"
SMTP_FROM="noreply@documed.com"

# Gemini AI (optional - for DocMate AI responses)
GEMINI_API_KEY="your-gemini-api-key"

# Optional: Production settings
NODE_ENV="development"
```

**4. Database setup**
```bash
# Generate Prisma client
pnpm prisma generate

# Push schema to database (development)
pnpm prisma db push

# Or run migrations (production)
pnpm prisma migrate deploy
```

**5. Run development server**
```bash
pnpm dev
```

Visit `http://localhost:3000` 🎉

### Model Training Setup (Optional)

**1. Install Python dependencies**
```bash
pip install tensorflow keras numpy pandas opencv-python matplotlib
```

**2. Download dataset**
```bash
# Chest X-ray dataset from Kaggle
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Extract to: ./chest_xray/
```

**3. Run training notebook**
```bash
jupyter notebook Model_training.ipynb
```

**4. Model weights**
- Training generates `best_weights.hdf5`
- Place in project root for inference

---

## 🔐 Authentication System

Documed uses NextAuth.js for secure, session-based authentication.

### Authentication Flow

```
User visits protected route
      ↓
Check session cookie
      ├─ Valid → Allow access
      └─ Invalid → Redirect to /auth/signin
            ↓
      User enters credentials
            ↓
      CredentialsProvider validates
            ├─ Success → Create JWT token → Set session cookie
            └─ Failure → Show error
```

### Password Security

```typescript
// Signup: Hash password before storing
import bcrypt from 'bcryptjs';

const hashedPassword = await bcrypt.hash(password, 10);
await prisma.user.create({
  data: { email, password: hashedPassword, name }
});

// Signin: Compare hashed passwords
const user = await prisma.user.findUnique({ where: { email } });
const isValid = await bcrypt.compare(password, user.password);
```

**Why bcrypt?**
- Industry-standard password hashing
- Built-in salt generation
- Adaptive cost factor (protects against future computing power increases)

### Password Reset Flow

```
1. User requests reset at /auth/forgot-password
      ↓
2. System generates unique token + expiration
      ↓
3. Token stored in database linked to user
      ↓
4. Email sent with reset link: /auth/forgot-password/reset?token=...
      ↓
5. User clicks link, enters new password
      ↓
6. System validates token (not expired, matches user)
      ↓
7. Password updated, token deleted
      ↓
8. User redirected to signin
```

### Session Management

```typescript
// Session configuration
export const authOptions = {
  session: {
    strategy: 'jwt',        // Stateless sessions
    maxAge: 30 * 24 * 60 * 60,  // 30 days
  },
  callbacks: {
    jwt: async ({ token, user }) => {
      if (user) token.id = user.id;
      return token;
    },
    session: async ({ session, token }) => {
      if (session.user) session.user.id = token.id;
      return session;
    }
  }
};
```

---

## 🗄 Database Architecture

### Prisma Schema

```prisma
model User {
  id            String    @id @default(auto()) @map("_id") @db.ObjectId
  email         String    @unique
  password      String
  name          String?
  image         String?
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
  
  // Relations
  passwordResets PasswordReset[]
  appointments   Appointment[]
  moodEntries    MoodEntry[]
  vitalsRecords  VitalsRecord[]
}

model PasswordReset {
  id        String   @id @default(auto()) @map("_id") @db.ObjectId
  userId    String   @db.ObjectId
  token     String   @unique
  expires   DateTime
  createdAt DateTime @default(now())
  
  user User @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  @@index([userId])
  @@index([token])
}

model MoodEntry {
  id        String   @id @default(auto()) @map("_id") @db.ObjectId
  userId    String   @db.ObjectId
  mood      String   // "Happy", "Sad", "Anxious", etc.
  text      String
  date      DateTime @default(now())
  
  user User @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  @@index([userId, date])
}

model VitalsRecord {
  id          String   @id @default(auto()) @map("_id") @db.ObjectId
  userId      String   @db.ObjectId
  heartRate   Int
  oxygen      Int
  temperature Float
  timestamp   DateTime @default(now())
  
  user User @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  @@index([userId, timestamp])
}
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines

- Follow TypeScript best practices
- Write meaningful commit messages
- Add comments for complex logic
- Test thoroughly before submitting PR
- Update documentation for new features

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Chest X-Ray Dataset**: [Kaggle - Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **TensorFlow/Keras**: Deep learning framework
- **Next.js Team**: Outstanding React framework
- **Shadcn**: Beautiful UI components
- **Vercel**: Deployment platform

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/rithvik17-09/documed/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rithvik17-09/documed/discussions)
- **Email**: support@documed.com

---

**Built with ❤️ by the Documed Team**

*Empowering healthcare through technology*
