# Quick Reference Guide

## Get Gemini API Key

Go to: https://aistudio.google.com/app/apikey

Click "Create API key" and copy the generated key.

## Environment Setup

### Windows (PowerShell)
```powershell
# Open .env file
notepad .env

# Add/Update these lines:
GEMINI_API_KEY="your-key-here"
NEXT_PUBLIC_GEMINI_API_KEY="your-key-here"

# Save and close
```

### Mac/Linux
```bash
# Edit .env
nano .env

# Add/Update these lines:
GEMINI_API_KEY="your-key-here"
NEXT_PUBLIC_GEMINI_API_KEY="your-key-here"

# Save: Ctrl+O, Enter, Ctrl+X
```

## Installation & Running

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Application will run on http://localhost:3001 (or 3000)
```

## Using X-Ray/MRI Analyzer

1. Open browser: `http://localhost:3001`
2. Login with your account
3. Navigate to "X-Ray Analyzer" module
4. Upload a medical image (JPG, PNG, or DICOM)
5. Click "Analyze Image"
6. Wait 2-10 seconds for Gemini API response
7. View results with confidence score and findings
8. Optional: Click "Export Report" to download analysis

## Troubleshooting Commands

```bash
# Check if dependencies are installed
npm list google-generative-ai

# Clear cache and reinstall
npm install --legacy-peer-deps

# Check environment variables (Windows PowerShell)
Get-Content .env

# Check environment variables (Linux/Mac)
cat .env

# Kill existing process and restart
# Windows: Ctrl+C in terminal, then: npm run dev
# Mac/Linux: Ctrl+C in terminal, then: npm run dev
```

## Testing with Sample Images

### For X-Ray Analysis
- Use chest X-ray images (PA or lateral views)
- JPG or PNG format
- Clear medical images work best
- Can be downloaded from:
  - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  - Or use any standard chest X-ray

### For MRI Analysis
- Use brain or spine MRI images
- JPG or PNG format (DICOM technically supported)
- Clear anatomical detail works best
- Can find samples from medical imaging databases

## API Response Examples

### Successful Analysis
```json
{
  "status": "Defective",
  "confidence": 92,
  "issues": [
    {
      "id": "1702641234567-abc123",
      "title": "Consolidation",
      "severity": "medium",
      "location": "Right lower lobe",
      "description": "Airspace consolidation suggests possible pneumonia"
    }
  ]
}
```

### Error Response
```json
{
  "error": "Failed to analyze image with Gemini API"
}
```

## Common Issues & Solutions

### Issue: "Missing Gemini API key"
**Solution:**
```bash
# Make sure .env has:
GEMINI_API_KEY="your-actual-key"

# Restart server (stop with Ctrl+C, run npm run dev again)
```

### Issue: "Failed to analyze image"
**Solutions:**
1. Check API key is valid and active
2. Verify image format is JPG/PNG
3. Try a different image
4. Check internet connection
5. Check Gemini API quota/limits

### Issue: Analysis takes too long
**Cause:** Network latency or Gemini API load
**Solution:** Be patient, wait up to 30 seconds
- First request may be slower
- Subsequent requests are typically faster

### Issue: "Method not allowed"
**Cause:** Using wrong HTTP method
**Solution:** Verify API endpoint only accepts POST requests

## File Locations

```
documed/
├── .env                                    # Your API keys
├── pages/
│   └── api/
│       └── analyze-image.ts               # NEW: Gemini API endpoint
├── components/
│   └── modules/
│       └── xray-analyser/
│           └── index.tsx                  # UPDATED: Gemini integration
├── package.json                           # UPDATED: Added dependency
├── GEMINI_SETUP_GUIDE.md                  # NEW: Detailed setup guide
└── IMPLEMENTATION_SUMMARY.md              # NEW: Implementation details
```

## Important Security Notes

⚠️ **Never commit .env to version control**
- Add to .gitignore if not already there
- Keep API keys private
- Rotate keys periodically in production

## Support Information

**API Used**: Google Generative AI (Gemini)
**Documentation**: https://ai.google.dev/docs
**API Limits**: Check your quota at https://aistudio.google.com/app/apikey

## Deployment Notes

For production deployment:
1. Use environment variables only (never hardcode keys)
2. Set up rate limiting on the API endpoint
3. Implement request validation
4. Add monitoring and logging
5. Use HTTPS for all connections
6. Consider caching analysis results
7. Add authentication to API endpoints

---

**Last Updated**: December 15, 2025
