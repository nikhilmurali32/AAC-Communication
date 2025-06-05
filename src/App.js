import React, { useState } from 'react';

const WORDS = [
  { text: 'No', color: '#FF6B6B' },
  { text: 'understand', color: '#FFD93D' },
  { text: 'I', color: '#6BCB77' },
  { text: 'happy', color: '#4D96FF' },
  { text: 'see', color: '#FF6F91' },
  { text: 'family', color: '#FFD6E0' },
  { text: 'yes', color: '#6A89CC' },
  { text: 'want', color: '#F7B801' },
  { text: 'help', color: '#A3A847' },
  { text: 'eat', color: '#FFB4A2' },
  { text: 'drink', color: '#B5EAD7' },
  { text: 'go', color: '#B2C7D9' },
  { text: 'play', color: '#F6D6AD' },
  { text: 'stop', color: '#FF8C42' },
  { text: 'sleep', color: '#A0CED9' },
  { text: 'pain', color: '#FFAAA7' },
  { text: 'school', color: '#B5EAD7' },
  { text: 'friend', color: '#C7CEEA' },
  { text: 'more', color: '#FFDAC1' },
  { text: 'like', color: '#E2F0CB' },
  { text: 'read', color: '#B5EAD7' },
  { text: 'music', color: '#C7CEEA' },
  { text: 'outside', color: '#FFB7B2' },
  { text: 'tired', color: '#B5EAD7' },
  { text: 'thanks', color: '#B2C7D9' }
];

function App() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState('');
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  const handleWordClick = (word) => {
    setInput((prev) => (prev ? prev + ' ' + word : word));
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    setLoading(true);
    setOutput('');

    try {
      const response = await fetch('http://localhost:5000/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: input.trim() })
      });

      if (!response.ok) {
        throw new Error('Failed to get response from server');
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setOutput(data.response);
      setHistory(prev => [...prev, { input: input.trim(), output: data.response }]);
    } catch (error) {
      console.error('Error:', error);
      setOutput('Sorry, there was an error processing your request. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleCloseOutput = () => {
    setOutput('');
    setInput('');
  };

  return (
    <div style={{ minHeight: '100vh', background: '#fff', padding: 0, margin: 0, fontFamily: 'Arial, sans-serif', position: 'relative', color: '#222' }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(5, 1fr)',
        gridTemplateRows: 'repeat(5, 1fr)',
        height: '100vh',
        width: '100vw',
        margin: 0,
        padding: 0,
      }}>
        {WORDS.map((word, idx) => (
          <button
            key={idx}
            onClick={() => handleWordClick(word.text)}
            style={{
              margin: 0,
              padding: 0,
              borderRadius: 0,
              border: '1px solid white',
              background: word.color,
              color: '#222',
              fontWeight: 'bold',
              fontSize: '1.1rem',
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxSizing: 'border-box',
              cursor: 'pointer',
            }}
          >
            {word.text}
          </button>
        ))}
      </div>

      {/* Input Bar */}
      <div style={{
        position: 'fixed',
        bottom: '5rem',
        left: '50%',
        transform: 'translateX(-50%)',
        width: '100%',
        display: 'flex',
        justifyContent: 'center',
        zIndex: 10,
        pointerEvents: 'none',
      }}>
        <div style={{
            background: input ? '#f5f5f7' : 'rgba(245,245,247,0.6)',
            borderRadius: '2rem',
            boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
            display: 'flex',
            alignItems: 'center',
            width: 'min(600px, 90vw)',
            padding: '0.5rem 1rem',
            pointerEvents: 'auto',
        }}>
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Tap words or type here..."
            style={{
              flex: 1,
              fontSize: '1.15rem',
              padding: '0.8rem 1rem',
              borderRadius: '2rem',
              border: 'none',
              outline: 'none',
              background: 'transparent',
              color: '#222',
              marginRight: '0.7rem',
            }}
          />
          <button
            onClick={handleSend}
            style={{
              background: '#4D96FF',
              color: '#fff',
              border: 'none',
              borderRadius: '50%',
              width: '2.5rem',
              height: '2.5rem',
              fontSize: '1.3rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 8px rgba(0,0,0,0.10)'
            }}
            aria-label="Send"
          >
            &#8593;
          </button>
        </div>
      </div>

      {/* Output/Loading */}
      {(loading || output) && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'rgba(255,255,255,0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 100,
          flexDirection: 'column',
        }}>
          {output && (
            <button
              onClick={handleCloseOutput}
              style={{
                position: 'absolute',
                top: '2rem',
                left: '2rem',
                background: 'rgba(0,0,0,0.08)',
                border: 'none',
                borderRadius: '50%',
                width: '2.5rem',
                height: '2.5rem',
                fontSize: '1.5rem',
                cursor: 'pointer',
                fontWeight: 'bold',
                zIndex: 101,
                color: '#222',
              }}
              aria-label="Close"
            >
              ×
            </button>
          )}
          <div style={{
            background: 'rgba(255,255,255,0.98)',
            borderRadius: '2rem',
            padding: '2.5rem 2.5rem',
            minWidth: '320px',
            minHeight: '120px',
            boxShadow: '0 4px 24px rgba(0,0,0,0.10)',
            textAlign: 'center',
            fontSize: '1.2rem',
            color: '#222',
            position: 'relative',
            maxWidth: '90vw',
            maxHeight: '60vh',
            overflowY: 'auto',
          }}>
            {loading ? (
              <div>
                <div className="loader" style={{ marginBottom: '1.2rem' }}>
                  <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="24" cy="24" r="20" stroke="#4D96FF" strokeWidth="5" strokeDasharray="31.4 31.4" strokeLinecap="round">
                      <animateTransform attributeName="transform" type="rotate" repeatCount="indefinite" dur="1s" from="0 24 24" to="360 24 24" />
                    </circle>
                  </svg>
                </div>
                <div>Processing...</div>
              </div>
            ) : (
              <div>{output}</div>
            )}
          </div>
        </div>
      )}

      {/* Floating History Button */}
      <button
        onClick={() => setShowHistory(prev => !prev)}
        style={{
          position: 'fixed',
          top: '1.5rem',
          right: '1.5rem',
          zIndex: 20,
          background: '#222',
          color: '#fff',
          border: 'none',
          borderRadius: '2rem',
          padding: '0.6rem 1.2rem',
          fontSize: '1rem',
          cursor: 'pointer',
          boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
        }}
      >
        {showHistory ? 'Close History' : 'Show History'}
      </button>

      {/* History Modal */}
      {showHistory && (
        <div style={{
          position: 'fixed',
          top: '4.5rem',
          right: '1.5rem',
          width: 'min(300px, 90vw)',
          maxHeight: '60vh',
          overflowY: 'auto',
          background: '#fff',
          borderRadius: '1rem',
          boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
          padding: '1rem',
          zIndex: 19,
        }}>
          <h3 style={{ marginTop: 0, fontSize: '1.1rem' }}>Conversation History</h3>
          {history.length === 0 && <p>No history yet.</p>}
          {history.slice().reverse().map((entry, index) => (
            <div key={index} style={{ marginBottom: '1rem' }}>
              <div style={{ fontSize: '0.9rem', marginBottom: '0.3rem', fontWeight: 'bold' }}>Input:</div>
              <div style={{ fontSize: '0.9rem', color: '#555', marginBottom: '0.3rem' }}>{entry.input}</div>
              <div style={{ fontSize: '0.9rem', marginBottom: '0.5rem' }}><strong>Response:</strong> {entry.output}</div>
              <button
                onClick={() => setInput(entry.input)}
                style={{
                  fontSize: '0.85rem',
                  background: '#4D96FF',
                  color: 'white',
                  border: 'none',
                  borderRadius: '0.5rem',
                  padding: '0.3rem 0.8rem',
                  cursor: 'pointer',
                }}
              >
                Reuse Input
              </button>
              <hr style={{ marginTop: '0.8rem', borderColor: '#eee' }} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
