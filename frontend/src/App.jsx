import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

// Utility to format seconds to MM:SS
function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

export default function App() {
  const [videoLink, setVideoLink] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [timestamps, setTimestamps] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setAnswer('');
    setTimestamps([]);
    try {
      const res = await axios.post('/query', { video_link: videoLink, question });
      setAnswer(res.data.answer);
      setTimestamps(res.data.timestamps);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1>YouTube RAG Query</h1>
      <form onSubmit={handleSubmit} className="form">
        <div className="form-group">
          <label>YouTube Video Link</label>
          <input
            type="text"
            value={videoLink}
            onChange={(e) => setVideoLink(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            required
          />
        </div>
        <div className="form-group">
          <label>Your Question</label>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. What do they say about AI?"
            required
          />
        </div>
        <button type="submit" disabled={loading} className="btn">
          {loading ? 'Processing...' : 'Submit'}
        </button>
      </form>

      {error && <div className="error">Error: {error}</div>}

      {answer && (
        <div className="result">
          <h2>Answer</h2>
          <p>{answer}</p>

          {timestamps.length > 0 && (
            <div className="timestamps">
              <h3>Relevant Timestamps:</h3>
              <ul>
                {timestamps.map((t, i) => (
                  <li key={i}>
                    <a
                      href={`${videoLink}&t=${Math.floor(t)}`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {formatTime(t)}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}