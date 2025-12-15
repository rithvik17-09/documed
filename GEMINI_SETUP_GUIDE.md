# X-Ray & MRI Gemini API Integration Setup Guide

## Overview
The X-Ray and MRI Analyzer modules have been updated to use Google's Gemini API for real-time image analysis. The AI model analyzes medical images and returns detailed findings.

## Changes Made

### 1. **New Backend API Endpoint** (`/pages/api/analyze-image.ts`)
- Accepts POST requests with base64 encoded image data
- Sends images to Gemini API for analysis
- Handles both MRI and X-Ray modes
- Returns structured analysis results with:
  - Status (Normal/Defective)
  - Confidence score (0-100%)
  - List of detected issues with severity levels

### 2. **Updated Frontend Component** (`/components/modules/xray-analyser/index.tsx`)
- Removed simulated analysis
- Integrated real Gemini API calls
- Converts image file to base64 before sending
- Displays real analysis results from the API

### 3. **Package Updates** (`package.json`)
- Added `google-generative-ai` package for Gemini API integration

### 4. **Environment Variables** (`.env`)
- Added `GEMINI_API_KEY` for server-side API calls
- Added `NEXT_PUBLIC_GEMINI_API_KEY` for client-side access

## Setup Instructions

### Step 1: Get Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key (or use an existing one)
3. Copy your API key

### Step 2: Update Environment Variables
Open `.env` file and replace the placeholder:
```env
GEMINI_API_KEY="your-actual-gemini-api-key-here"
NEXT_PUBLIC_GEMINI_API_KEY="your-actual-gemini-api-key-here"
```

### Step 3: Install Dependencies
Run the following command to install the new package:
```bash
npm install
# or
pnpm install
```

### Step 4: Run the Application
```bash
npm run dev
# or
pnpm dev
```

## How It Works

### Image Upload Flow
1. User uploads an image (JPG, PNG, or DICOM format)
2. Image is displayed as preview
3. User clicks "Analyze Image" button
4. Component converts image to base64 format
5. Sends POST request to `/api/analyze-image` with:
   - Base64 encoded image data
   - Image MIME type
   - Analysis mode (mri or xray)

### Backend Processing
1. API receives the request
2. Initializes Gemini AI model
3. Sends image + specialized prompt to Gemini
4. Gemini analyzes the medical image
5. Parses JSON response from Gemini
6. Returns structured results to frontend

### Result Display
1. Confidence score displayed prominently
2. Status shown (Normal/Defective)
3. Detected issues listed with:
   - Title
   - Severity level (Low/Medium/High)
   - Anatomical location
   - Detailed description
4. Export Report functionality available

## API Request/Response Format

### Request
```json
{
  "imageData": "base64_encoded_image_string",
  "imageType": "image/jpeg",
  "mode": "xray" // or "mri"
}
```

### Response
```json
{
  "status": "Defective",
  "confidence": 95,
  "issues": [
    {
      "id": "unique-id",
      "title": "Fracture",
      "severity": "high",
      "location": "Left rib",
      "description": "Cortical discontinuity noted consistent with acute fracture"
    }
  ]
}
```

## Testing

### Test X-Ray Analysis
1. Open the application and navigate to the X-Ray Analyzer module
2. Click the "X-ray" button to switch to X-ray mode
3. Upload a chest X-ray image
4. Click "Analyze Image"
5. View results

### Test MRI Analysis
1. Click the "MRI" button to switch to MRI mode
2. Upload an MRI scan image
3. Click "Analyze Image"
4. View results

## Important Notes

⚠️ **Medical Disclaimer**
- This tool is for educational and research purposes only
- Results are AI-generated and should NOT be used for actual medical diagnosis
- Always consult qualified medical professionals for diagnosis and treatment decisions
- The disclaimer is prominently displayed on the UI

## Troubleshooting

### "Missing Gemini API key" Error
- Check that both `GEMINI_API_KEY` and `NEXT_PUBLIC_GEMINI_API_KEY` are set in `.env`
- Make sure the API key is valid and not expired
- Restart the development server after updating `.env`

### "Failed to analyze image" Error
- Ensure the image format is supported (JPG, PNG, DICOM)
- Check that the file size is reasonable
- Verify your Gemini API key has quota available
- Check browser console for detailed error messages

### Image Not Processing
- Make sure the image is selected (preview appears before analyzing)
- Check network tab in browser DevTools
- Verify API endpoint is reachable at `/api/analyze-image`

## Security Considerations

1. **API Key Protection**: Use `GEMINI_API_KEY` server-side for sensitive operations
2. **CORS**: API endpoint is only callable from your Next.js application
3. **Rate Limiting**: Consider implementing rate limiting for production
4. **Image Storage**: Images are not stored; they are analyzed and discarded
5. **Privacy**: Process images on-the-fly without persistent storage

## Performance Notes

- Initial image analysis may take 2-10 seconds depending on Gemini API response time
- Larger images may take longer to process
- Consider showing loading state to users during analysis

## Future Enhancements

Potential improvements:
1. Add image preprocessing for better quality
2. Implement caching for similar images
3. Add batch analysis capability
4. Implement usage analytics and monitoring
5. Add error retry logic with exponential backoff
6. Support for more medical image formats (DICOM with proper parsing)

---

**Last Updated**: December 15, 2025
**Status**: Ready for Testing
