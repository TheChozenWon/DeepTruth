# 🔍 DeepTruth - AI-Powered Fact-Checking Platform

<div align="center">
  <img src="Chrome Extension/images/icon128.png" alt="DeepTruth Logo" width="128" height="128">
  <p><em>Unveiling Truth in the Digital Age</em></p>
</div>

![image](https://github.com/user-attachments/assets/6ab275ae-9766-4517-bccc-764b51053b26)

Frontend Link: https://github.com/jaydoshi2/Deeptruth-Frontend
Backend Link: https://github.com/jaydoshi2/Deeptruth-Backend
## 🌟 Overview

DeepTruth is an innovative fact-checking platform that combines the power of Google's Gemini AI and DistilBERT to provide real-time analysis of news articles and claims. Our Chrome extension makes it easy to verify information while browsing the web.

### 🎯 Key Features

- **Real-time Analysis**: Instant fact-checking of any article title
- **Dual AI Model**: Combines Gemini AI and DistilBERT for enhanced accuracy
- **Smart Weighting**: 70% Gemini, 30% DistilBERT for optimal results
- **Chrome Extension**: Seamless integration with your browsing experience
- **Detailed Reports**: Comprehensive analysis with confidence scores
- **Source Verification**: Cross-references multiple news sources

## 🛠️ Tech Stack

- **Frontend**: React + Vite
- **Backend**: Django + Django REST Framework
- **AI Models**: 
  - Google Gemini Pro
  - Fine-tuned DistilBERT
- **Database**: MongoDB
- **Browser Extension**: Chrome Extension API

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB
- Chrome Browser
- Google Gemini API Key

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deeptruth.git
cd deeptruth/backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

5. Run migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

6. Start the server:
```bash
python manage.py runserver
```

### Frontend Setup

1. Navigate to the Deeptruth directory:
```bash
cd ../Deeptruth
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev -- --host
```

### Chrome Extension Setup

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" and select the `Chrome Extension` directory
4. The DeepTruth icon should appear in your Chrome toolbar

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory with:

```env
GEMINI_API_KEY=your_gemini_api_key
MONGODB_URI=your_mongodb_uri
DEBUG=True
```

## 📊 API Endpoints

- `POST /api/verify-claim/`: Verify an article title
- `GET /api/false-news/`: List all analyzed claims

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini AI Team
- Hugging Face for DistilBERT
- Django and React communities
- All contributors and supporters

## 📞 Support

For support, email support@deeptruth.com or join our Slack channel.

---

<div align="center">
  <p>Made with ❤️ by the DeepTruth Team</p>
  <p>© 2024 DeepTruth. All rights reserved.</p>
</div>
