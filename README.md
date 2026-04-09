================================================================================
                        DermAI — PROJECT DETAILS
   Intelligent Skin Disease Early Screening System Powered by AI/ML
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PROJECT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Project Name    : DermAI — Intelligent Skin Disease Early Screening System
Domain          : AI/ML + Full-Stack Web Development (Healthcare)
Category        : Deep Learning-based Medical Image Classification
Purpose         : Early screening and detection of skin diseases using AI-powered
                  image analysis combined with symptom-based refinement.

DermAI is a full-stack AI-powered web application that enables users to upload
images of skin conditions and receive an AI-driven preliminary diagnosis. The system
uses a deep learning model (MobileNetV2) trained on dermatological images to classify
skin conditions into 8 categories. After the initial image-based prediction, the
system further refines the diagnosis through an intelligent symptom-based questioning
system, significantly boosting overall accuracy and clinical relevance.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. DISEASE CLASSES (8 Categories)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The model classifies skin images into the following 8 categories:

  1. Eczema        — Chronic inflammatory skin condition (itchy, red, dry patches)
  2. Keratosis     — Rough, scaly skin growths (actinic/seborrheic keratosis)
  3. Nevi          — Moles / benign melanocytic nevi
  4. Normal        — Healthy skin, no disease detected
  5. Psoriasis     — Autoimmune condition with thick, red, scaly patches
  6. SkinCancer    — Malignant skin growths (melanoma, basal/squamous cell carcinoma)
  7. Tinea         — Fungal infection of the skin (ringworm, athlete's foot)
  8. Warts         — HPV-induced benign skin growths

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. AI/ML MODEL — DEVELOPMENT JOURNEY & TECHNICAL DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3.1 MODEL EXPERIMENTATION HISTORY
─────────────────────────────────

We experimented with multiple deep learning architectures before arriving at
the final model. The journey was as follows:

  ┌──────────────────────┬──────────────────┬───────────────────────────────────┐
  │ Model Architecture   │ Accuracy Achieved│ Remarks                           │
  ├──────────────────────┼──────────────────┼───────────────────────────────────┤
  │ EfficientNet         │ ~11%             │ Extremely poor. The model failed  │
  │                      │                  │ to converge — likely due to the   │
  │                      │                  │ architecture being too heavy for  │
  │                      │                  │ the dataset size and quality.     │
  ├──────────────────────┼──────────────────┼───────────────────────────────────┤
  │ ResNet               │ ~40%             │ Moderate improvement, but still   │
  │                      │                  │ underfitting. ResNet's deeper     │
  │                      │                  │ architecture overfitted on noisy  │
  │                      │                  │ data and could not generalize.    │
  ├──────────────────────┼──────────────────┼───────────────────────────────────┤
  │ MobileNetV2 (Initial)│ ~60%             │ Best results so far. Lightweight  │
  │                      │                  │ architecture suited the dataset.  │
  │                      │                  │ But data quality was still a      │
  │                      │                  │ major bottleneck.                 │
  ├──────────────────────┼──────────────────┼───────────────────────────────────┤
  │ MobileNetV2 (Final)  │ Training: ~80%   │ After extensive data and training │
  │                      │ Testing:  ~85%   │ optimizations (see below), we     │
  │                      │                  │ achieved the final accuracy.      │
  └──────────────────────┴──────────────────┴───────────────────────────────────┘

3.2 WHY MobileNetV2?
─────────────────────

MobileNetV2 was chosen as the final architecture because:

  • Lightweight — Fewer parameters (~3.4M) compared to ResNet (~25M) and
    EfficientNet (~5-7M), making it faster to train and deploy.
  • Efficient — Uses depthwise separable convolutions and inverted residual
    blocks, ideal for limited compute resources.
  • Transfer Learning Ready — Pre-trained on ImageNet (1.2M images, 1000 classes)
    which provides strong feature extraction even with our limited dataset.
  • Best Fit — Among all architectures tested, MobileNetV2 showed the best
    learning curve and convergence behavior on our dermatological dataset.

3.3 KEY OPTIMIZATIONS THAT IMPROVED ACCURACY (60% → 85%)
─────────────────────────────────────────────────────────

The initial MobileNetV2 only reached ~60% because the dataset quality was poor.
We applied the following critical optimizations:

  A) DATA AUGMENTATION
     • Rotation Range: ±20°
     • Zoom Range: 0.2
     • Width/Height Shift: ±10%
     • Horizontal Flip: Enabled
     • Brightness Range: [0.8, 1.2]
     → Purpose: Artificially increase dataset diversity and reduce overfitting
       by exposing the model to varied perspectives of the same images.

  B) TRANSFER LEARNING WITH ImageNet WEIGHTS
     • Used MobileNetV2 pre-trained on ImageNet as the base feature extractor.
     • The base model was initially FROZEN (all layers non-trainable) during
       Phase 1 training to preserve learned features.
     → Purpose: Leverage pre-learned visual features (edges, textures, shapes)
       from 1.2 million diverse images.

  C) TWO-PHASE TRAINING (FREEZE → FINE-TUNE)
     • Phase 1 — Frozen Base (10 epochs):
       Base model weights frozen. Only the custom classification head was trained.
       Optimizer: Adam (lr=1e-4)
     • Phase 2 — Fine-Tuning (15 epochs):
       Unfroze the last 80 layers of MobileNetV2 for fine-tuning.
       Optimizer: Adam (lr=1e-5, much smaller to prevent catastrophic forgetting)
     → Purpose: First learn the classification task, then fine-tune the feature
       extractor for domain-specific dermatological patterns.

  D) AdamW / Adam OPTIMIZER
     • Used Adam optimizer with carefully tuned learning rates.
     • Phase 1: lr = 1e-4 (faster convergence for head training)
     • Phase 2: lr = 1e-5 (gentle fine-tuning of base layers)

  E) LABEL SMOOTHING
     • Applied CategoricalCrossentropy with label_smoothing = 0.1
     → Purpose: Prevents the model from becoming overconfident on training data,
       improving generalization to unseen test images.

  F) REGULARIZATION (Dropout + BatchNorm)
     • Dropout 0.6 after first Dense layer (512 units)
     • Dropout 0.4 after second Dense layer (256 units)
     • Batch Normalization between Dense layers
     → Purpose: Prevent overfitting on the limited training data.

  G) EARLY STOPPING & LEARNING RATE SCHEDULING
     • EarlyStopping: patience=5, restore_best_weights=True
       (Stop training if validation accuracy doesn't improve for 5 epochs)
     • ReduceLROnPlateau: factor=0.3, patience=2, min_lr=1e-6
       (Reduce learning rate by 70% if validation loss plateaus)
     • ModelCheckpoint: Save best model (by val_accuracy) as best_model.keras
     → Purpose: Automatically find the best training point and prevent
       overfitting by stopping at the right time.

  H) DATASET CLEANING & CURATION
     • Removed duplicate, blurry, and mislabeled images
     • Ensured proper class distribution
     • Standardized image quality across classes

3.4 MODEL ARCHITECTURE SUMMARY
───────────────────────────────

  Input: 224 × 224 × 3 (RGB Image)
    ↓
  MobileNetV2 Base (ImageNet pre-trained, last 80 layers fine-tuned)
    ↓
  GlobalAveragePooling2D
    ↓
  Dense(512, ReLU) → BatchNormalization → Dropout(0.6)
    ↓
  Dense(256, ReLU) → Dropout(0.4)
    ↓
  Dense(8, Softmax)  ← 8 skin disease classes
    ↓
  Output: Probability distribution over 8 classes

3.5 FINAL MODEL ACCURACY
─────────────────────────

  • Training Accuracy : ~80%
  • Testing Accuracy  : ~85%

  The testing accuracy being higher than training accuracy is a positive sign,
  indicating that the model generalizes well to unseen data WITHOUT overfitting.
  This is largely due to data augmentation (which makes training harder) and
  dropout regularization.

3.6 SYMPTOM-BASED REFINEMENT SYSTEM (Accuracy Booster)
───────────────────────────────────────────────────────

To further increase diagnostic accuracy beyond the image-only prediction,
we implemented an intelligent Symptom Verification System:

  • After the image is analyzed by the AI model, the system asks the user
    6 targeted symptom questions:
      1. Are you experiencing itching?
      2. Is there visible redness or inflammation?
      3. Do you feel a burning or stinging sensation?
      4. Do you have a history of allergies?
      5. Is the affected area spreading?
      6. Is the skin dry, flaky, or scaly?

  • Each disease has a predefined symptom profile (e.g., Eczema = itching + redness
    + allergy + dry/flaky). The system compares user responses against these profiles.

  • Matching symptoms BOOST the confidence by 15% (× 1.15)
  • Mismatching symptoms REDUCE the confidence by 10% (× 0.90)

  • Confidences are then normalized (sum to 1.0) and diseases are re-ranked.

  • Final labels are assigned:
      ≥ 60% → "Highly Likely"
      ≥ 30% → "Possible"
      < 30% → "Maybe"

  → This hybrid approach (Image AI + Symptom Logic) significantly improves
    the quality and reliability of the final diagnosis compared to image-only
    prediction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. SYSTEM ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                           USER (Browser)                              │
  │                                                                       │
  │  ┌─────────────────┐    ┌──────────────────────────────────────────┐   │
  │  │  Landing Page    │    │           Dashboard                      │   │
  │  │  (Hero, About,   │    │  • Diagnostic Scanner (Upload + Scan)   │   │
  │  │   Timeline,      │    │  • Symptom Verification Terminal        │   │
  │  │   Particles 3D)  │    │  • AI Results with Confidence Rings     │   │
  │  │                  │    │  • Scan History + PDF Reports           │   │
  │  │  [Next.js 16]    │    │  • Find Dermatologists (Nearby)         │   │
  │  │  [React 19]      │    │  • User Profile                        │   │
  │  │  [TailwindCSS 4] │    │  • Firebase Authentication             │   │
  │  └─────────────────┘    └──────────────────────────────────────────┘   │
  └───────────────────────────────┬────────────────────────────────────────┘
                                  │ HTTP REST API (JSON + FormData)
                                  ↓
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                      BACKEND (FastAPI + Python)                        │
  │                                                                       │
  │  Endpoints:                                                           │
  │  • POST /predict        → Image upload → MobileNetV2 prediction       │
  │  • POST /symptom-check  → Symptom refinement → Final diagnosis        │
  │  • POST /refine         → Alias for symptom-check (without history)   │
  │  • GET  /history        → Fetch user scan history                     │
  │  • GET  /derma          → Fetch nearby dermatologists                 │
  │                                                                       │
  │  Model: best_model.keras (MobileNetV2, ~19 MB)                       │
  │  Server: Uvicorn ASGI on port 8000                                    │
  └─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. TECHNOLOGY STACK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5.1 FRONTEND (mainfrontend/)
────────────────────────────

  Framework      : Next.js 16.1.6 (React 19, App Router, Turbopack)
  Language       : TypeScript / TSX
  Styling        : TailwindCSS v4 (custom design system)
  UI Library     : Lucide React (icons), Framer Motion (animations)
  3D Graphics    : Three.js + React Three Fiber + Drei + Postprocessing
                   → Used for the DNA helix particle animation on the landing page
  Authentication : Firebase Auth (Google Sign-In, Email/Password)
  PDF Generation : jsPDF (client-side PDF report generation)
  Fonts          : Orbitron, Inter, JetBrains Mono, Outfit (Google Fonts)
  Design Theme   : "Deep Space Medical" — dark, futuristic, neon-cyan accents

5.2 BACKEND (backend/)
──────────────────────

  Framework      : FastAPI (Python)
  Server         : Uvicorn (ASGI)
  ML Framework   : TensorFlow / Keras
  Image Processing: Pillow (PIL)
  Numerical Ops  : NumPy
  Data Storage   : JSON file (scan_history.json) — in-memory + file persistence
  CORS           : Enabled for localhost ports (3000, 5173-5175)

5.3 AI/ML
─────────

  Model          : MobileNetV2 (Transfer Learning from ImageNet)
  Training Script: mobilenetV2_model_training.py
  Saved Model    : best_model.keras (~19 MB)
  Input Size     : 224 × 224 × 3 (RGB)
  Output         : 8-class softmax probability distribution
  Training Tools : TensorFlow, Keras, sklearn, matplotlib, seaborn

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. FRONTEND FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6.1 LANDING PAGE
────────────────

  • Hero Section — Full-screen with animated text, glowing CTA button
  • 3D DNA Particle Background — Interactive Three.js particle system that
    morphs from a DNA helix (at top) to scattered particles (on scroll)
  • About Section — Three feature cards with hover effects
  • Info Section — Detailed system information
  • Timeline — Stages of skin disease progression
  • Custom Cursor — Neon-cyan animated cursor with hover effects
  • Responsive Navbar — Transparent → blur-on-scroll, mobile hamburger menu

6.2 AUTHENTICATION
──────────────────

  • Firebase Authentication (Login/Signup modal)
  • Google Sign-In support
  • Email/Password registration
  • Persistent session via Firebase auth state
  • User data stored in localStorage for dashboard access

6.3 DASHBOARD
─────────────

  • Sidebar Navigation — Sidebar with icons for Scanner, History, Profile,
    Find Doctors
  • Diagnostic Scanner — Drag-and-drop image upload → AI scan with animated
    progress → Symptom verification terminal → Results with confidence rings
  • Scan History — All past scans with disease name, confidence percentage,
    date/time, and PDF report download button for each entry
  • PDF Report — Full diagnostic report generated client-side with jsPDF
    (includes primary diagnosis, model predictions, symptom assessment,
    refined results, and disclaimer)
  • Find Doctors — Nearby dermatologists list with distance, rating, phone
  • User Profile — Displays user name, email, and account info
  • Sparkling star background effect in dashboard

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. BACKEND API ENDPOINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌────────────────────┬────────┬──────────────────────────────────────────────┐
  │ Endpoint           │ Method │ Description                                  │
  ├────────────────────┼────────┼──────────────────────────────────────────────┤
  │ /predict           │ POST   │ Upload skin image → Returns top 3 disease    │
  │                    │        │ predictions with confidence scores           │
  │                    │        │ Input: FormData (file + user_id)             │
  │                    │        │ Output: { predictions, questions }           │
  ├────────────────────┼────────┼──────────────────────────────────────────────┤
  │ /symptom-check     │ POST   │ Symptom-based refinement + history save      │
  │                    │        │ Input: { predictions, symptoms, user_id }    │
  │                    │        │ Output: { results } with refined scores      │
  ├────────────────────┼────────┼──────────────────────────────────────────────┤
  │ /refine            │ POST   │ Same as /symptom-check but without saving    │
  │                    │        │ to history (legacy endpoint)                 │
  ├────────────────────┼────────┼──────────────────────────────────────────────┤
  │ /history           │ GET    │ Fetch scan history for a user                │
  │                    │        │ Query: ?user_id=<uid>                        │
  │                    │        │ Output: { history: [...] }                   │
  ├────────────────────┼────────┼──────────────────────────────────────────────┤
  │ /derma             │ GET    │ Fetch nearby dermatologists                  │
  │                    │        │ Query: ?lat=<lat>&lng=<lng>                  │
  │                    │        │ Output: { doctors: [...] }                   │
  └────────────────────┴────────┴──────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. PROJECT DIRECTORY STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  final_AIML_part/
  │
  ├── backend/                          ← FastAPI Backend
  │   ├── api.py                        ← Main API with all endpoints
  │   ├── best_model.keras              ← Trained MobileNetV2 model (~19 MB)
  │   ├── classes.json                  ← Class label mapping
  │   ├── scan_history.json             ← Persistent scan history storage
  │   └── requirements.txt              ← Python dependencies
  │
  ├── mainfrontend/                     ← Next.js 16 Frontend (Primary)
  │   └── src/
  │       ├── app/
  │       │   ├── page.tsx              ← Landing page (Home)
  │       │   ├── layout.tsx            ← Root layout with fonts & auth
  │       │   ├── globals.css           ← Global styles & design tokens
  │       │   └── dashboard/
  │       │       └── page.tsx          ← Dashboard page
  │       ├── components/
  │       │   ├── Hero/Hero.tsx         ← Hero section with animated text
  │       │   ├── About/About.tsx       ← About section with feature cards
  │       │   ├── Navbar/Navbar.tsx     ← Navigation bar with auth
  │       │   ├── ParticleScene/        ← 3D DNA particle animation
  │       │   ├── Timeline/             ← Skin disease stages timeline
  │       │   ├── InfoSection/          ← Information section
  │       │   ├── Footer/              ← Footer component
  │       │   ├── Auth/AuthModal.tsx    ← Login/Signup modal (Firebase)
  │       │   ├── CustomCursor.tsx     ← Custom animated cursor
  │       │   └── Dashboard/
  │       │       ├── DiagnosticScanner.tsx  ← Image upload + AI scan
  │       │       ├── HistoryPanel.tsx       ← Scan history + PDF download
  │       │       ├── FindDoctors.tsx        ← Nearby dermatologists
  │       │       ├── ProfilePanel.tsx       ← User profile
  │       │       └── Sidebar.tsx            ← Dashboard sidebar navigation
  │       ├── context/
  │       │   └── AuthContext.tsx       ← Firebase auth context provider
  │       ├── lib/
  │       │   └── firebase.ts          ← Firebase configuration
  │       └── utils/
  │           └── buildPDF.ts          ← Shared PDF report generator
  │
  ├── mobilenetV2_model_training.py     ← Model training script
  ├── classes.json                      ← Class indices mapping
  ├── runguide.txt                      ← How to run the project
  └── PROJECT_DETAILS.txt               ← This file
  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. HOW TO RUN THE PROJECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  STEP 1: Start the Backend (FastAPI)
  ────────────────────────────────────
    cd backend
    pip install -r requirements.txt
    python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload

    → Backend runs on: http://127.0.0.1:8000
    → API docs available at: http://127.0.0.1:8000/docs

  STEP 2: Start the Frontend (Next.js)
  ─────────────────────────────────────
    cd mainfrontend
    npm install
    npm run dev

    → Frontend runs on: http://localhost:3000

  STEP 3: Use the Application
  ────────────────────────────
    1. Open http://localhost:3000 in browser
    2. Sign up / Login via the navbar
    3. Go to Dashboard
    4. Upload a skin image in Diagnostic Scanner
    5. Answer the 6 symptom questions
    6. View the AI diagnosis results
    7. Download PDF report
    8. View scan history with downloadable reports

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10. KEY TECHNICAL HIGHLIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✅ Transfer Learning with MobileNetV2 (ImageNet pre-trained weights)
  ✅ Two-Phase Training Strategy (Freeze → Fine-Tune last 80 layers)
  ✅ Extensive Data Augmentation (rotation, zoom, shift, flip, brightness)
  ✅ Label Smoothing (0.1) to improve generalization
  ✅ Dropout Regularization (0.6 + 0.4) to prevent overfitting
  ✅ Batch Normalization for training stability
  ✅ Early Stopping + Learning Rate Scheduling for optimal convergence
  ✅ Hybrid Diagnosis: Image AI + Symptom-based Refinement System
  ✅ Real-time 3D Particle Animation (Three.js + React Three Fiber)
  ✅ Firebase Authentication (Google + Email/Password)
  ✅ Client-side PDF Report Generation (jsPDF)
  ✅ RESTful API with FastAPI + CORS
  ✅ Persistent Scan History (JSON file storage)
  ✅ Responsive Design (Mobile + Desktop)
  ✅ Modern "Deep Space Medical" UI Theme with neon-cyan accents

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
11. DEPENDENCIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Backend (Python):
    • fastapi           — Web framework for API
    • uvicorn           — ASGI server
    • tensorflow        — Deep learning framework
    • pillow            — Image processing
    • numpy             — Numerical operations
    • python-multipart  — File upload handling

  Frontend (Node.js):
    • next 16.1.6       — React framework (App Router + Turbopack)
    • react 19.2.3      — UI library
    • tailwindcss v4    — CSS framework
    • framer-motion     — Animation library
    • three.js          — 3D graphics
    • @react-three/fiber, drei, postprocessing — Three.js React bindings
    • firebase          — Authentication
    • jspdf             — PDF generation
    • lucide-react      — Icon library
    • typescript        — Type safety

  Training (Python):
    • tensorflow/keras  — Model training
    • scikit-learn      — Classification report & confusion matrix
    • matplotlib        — Training visualization
    • seaborn           — Confusion matrix heatmap

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
12. DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DermAI is an AI-powered SCREENING tool designed for INFORMATIONAL PURPOSES ONLY.
  It is NOT a replacement for professional medical diagnosis. Users should always
  consult a qualified dermatologist for proper diagnosis and treatment.

================================================================================
                        End of DermAI Project Details
================================================================================
