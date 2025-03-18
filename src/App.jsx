import axios from 'axios';

// Get the API URL from environment or use default
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function App() {
  const handleGoClick = async () => {
    setLoading(true);
    setGoClicked(true);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/api/verify-claim/`, {
        article_title: searchText
      });
      // ... rest of the code ...
    } catch (err) {
      console.error('Error posting data:', err);
      setError(err.response?.data?.error || 'An error occurred while verifying the claim');
    } finally {
      setLoading(false);
    }
  };

  // ... rest of the component ...
} 
