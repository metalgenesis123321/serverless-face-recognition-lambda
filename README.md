# Serverless Face Recognition Pipeline on AWS Lambda

A fully serverless face recognition system built using AWS Lambda and SQS. The pipeline implements a two-stage processing approach: face detection using MTCNN followed by face recognition using FaceNet (InceptionResnetV1), all running on AWS Lambda functions.

## Architecture Overview

```
+--------+     +-------------------+     +-----------+     +--------------------+     +-----------+
| Client | --> | Face Detection    | --> | SQS Queue | --> | Face Recognition   | --> | SQS Queue |
|        |     | Lambda (MTCNN)    |     | (Request) |     | Lambda (FaceNet)   |     | (Response)|
+--------+     +-------------------+     +-----------+     +--------------------+     +-----------+
```

## Components

### Face Detection Lambda

The first stage of the pipeline that detects and extracts faces from input images.

**Technology:** MTCNN (Multi-task Cascaded Convolutional Networks) from facenet-pytorch

**Responsibilities:**
- Receive base64-encoded images via API Gateway
- Detect faces in the input image
- Extract and normalize the detected face region
- Encode the cropped face as base64
- Send to SQS request queue for recognition

**Input Format:**
```json
{
  "content": "<base64-encoded-image>",
  "request_id": "<unique-identifier>",
  "filename": "<original-filename>"
}
```

### Face Recognition Lambda

The second stage that identifies the detected face against a pre-trained embedding database.

**Technology:** InceptionResnetV1 (FaceNet) pretrained on VGGFace2

**Responsibilities:**
- Triggered by SQS request queue messages
- Decode the cropped face image
- Generate face embedding using FaceNet
- Compare against stored embeddings using Euclidean distance
- Return the closest matching identity
- Send results to SQS response queue

**Recognition Process:**
1. Load pre-computed embeddings from weight file
2. Generate embedding for input face
3. Calculate distance to all stored embeddings
4. Return identity with minimum distance

## AWS Services Used

| Service | Purpose |
|---------|---------|
| Lambda | Hosts face detection and recognition functions |
| API Gateway | HTTP endpoint for face detection Lambda |
| SQS | Decouples detection and recognition stages |

## Technical Details

**Face Detection Model:**
- MTCNN with image size 240x240
- Margin: 0, Min face size: 20 pixels

**Face Recognition Model:**
- InceptionResnetV1 pretrained on VGGFace2
- Embedding dimension: 512
- Distance metric: Euclidean distance

**Lambda Configuration:**
- Models initialized outside handler for warm start optimization
- Long-running inference tasks benefit from Lambda's execution environment reuse

## Prerequisites

- AWS Account with Lambda and SQS permissions
- Pre-computed face embeddings file (`resnetV1_video_weights.pt`)
- Lambda layers or container images with:
  - PyTorch
  - facenet-pytorch
  - NumPy
  - Pillow

## Dependencies

```
torch
facenet-pytorch
numpy
Pillow
boto3
```

## Deployment

1. Create Lambda layers with required dependencies
2. Package and deploy face detection Lambda
3. Package and deploy face recognition Lambda with weights file
4. Create SQS queues (request and response)
5. Configure API Gateway trigger for detection Lambda
6. Configure SQS trigger for recognition Lambda

## Usage

```bash
# Encode image to base64
IMAGE_B64=$(base64 -w 0 face.jpg)

# Send request
curl -X POST https://<api-gateway-url>/detect \
  -H "Content-Type: application/json" \
  -d '{
    "content": "'$IMAGE_B64'",
    "request_id": "req-001",
    "filename": "face.jpg"
  }'
```

## Performance Optimization

The implementation includes several optimizations for Lambda:

- **Cold Start Mitigation:** Models are initialized at module level, outside the handler function
- **Memory Efficiency:** Images are processed in-memory using BytesIO buffers
- **Batch Processing:** SQS triggers can be configured to batch messages

## Project Context

This project was developed as part of a Cloud Computing course at Arizona State University, demonstrating serverless architecture patterns using AWS Lambda, event-driven processing with SQS, and deployment of machine learning models in a Function-as-a-Service (FaaS) environment.
