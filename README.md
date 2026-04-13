# DermAI -- AI Skin Disease Screening System (DTI Project)

DermAI is an AI-powered skin disease screening system developed as part
of our DTI (Design Thinking & Innovation) project. The system leverages
computer vision and deep learning techniques to analyze skin conditions
and provide real-time, preliminary insights.

The goal of this project is to make early-stage skin assessment more
accessible and efficient by combining AI with a user-friendly interface.
By enabling quick analysis through images and intelligent reasoning,
DermAI assists users in understanding potential skin conditions and
encourages timely medical consultation.

------------------------------------------------------------------------

## What problem it solves

Skin diseases are often ignored in their early stages due to lack of
awareness or limited access to dermatologists. Traditional diagnosis can
be time-consuming and not easily accessible to everyone.

DermAI addresses this issue by providing an AI-based system that allows
users to scan their skin and receive instant analysis. This helps in
early detection, awareness, and better healthcare decision-making.

------------------------------------------------------------------------

## Key features

-   AI-based skin detection using deep learning\
-   Real-time prediction system\
-   Symptom-based refinement system\
-   Hybrid decision logic (image + symptoms)\
-   History tracking of results\
-   Scalable AI pipeline

------------------------------------------------------------------------

## Architecture & technology (summary)

-   AI Models: MobileNetV2 (final), experimented with ResNet and
    EfficientNet\
-   Data Pipeline: Collection, preprocessing, augmentation, balancing\
-   Backend: Model inference & data flow\
-   Frontend: User interaction and image input\
-   Decision Layer: Combines AI + symptom reasoning

------------------------------------------------------------------------

## My Personal Contribution

I was primarily responsible for the core AI development and overall
system integration.

I handled the complete data pipeline, including dataset collection,
preprocessing (cleaning, resizing, normalization), class balancing, and
augmentation. I also addressed class imbalance using appropriate
techniques.

I experimented with EfficientNet, ResNet, and MobileNetV2, and selected
MobileNetV2 based on performance and efficiency. I worked on
hyperparameter tuning, loss optimization, and improving accuracy.

I also designed a symptom-based reasoning system that refines
predictions using user input. Additionally, I built a hybrid decision
system combining image predictions with symptom logic.

Finally, I integrated the AI model into the pipeline, enabling a
complete end-to-end workflow from input to prediction.

------------------------------------------------------------------------

## Impact & goals

DermAI aims to make early skin disease detection faster and more
accessible. The goal is to evolve it into a scalable healthcare support
system.

------------------------------------------------------------------------

# 🚀 How to Run the Project

## Clone the repository

git clone
https://github.com/manasnigam11/DermAI---AI-Skin-Disease-Screening-System.git

cd DermAI---AI-Skin-Disease-Screening-System

## Install dependencies

pip install -r requirements.txt

## Run project

How to run backend :

cd backend

python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload

How to run Frontend :

cd frontend

npm install

npm run dev

Make sure backend is running before starting the frontend.

------------------------------------------------------------------------

## Contribution in this project

Vivek Nath Tiwari - All responsibilities related to frontend

Manas Nigam - All responsibilities related to AIML part and symptom-based reasoning system

Pratyush Dev - All responsibilities related to Backend
