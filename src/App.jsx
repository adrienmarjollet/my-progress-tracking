import React from 'react';
import './styles/MessageDisplay.css';
import MessageDisplay from './components/MessageDisplay';

function App() {
  // ...existing code...

  // When rendering messages, use the new MessageDisplay component
  return (
    <div className="app-container">
      {/* ...existing code... */}
      <div className="messages-container">
        {messages.map((message, index) => (
          <MessageDisplay key={index} message={message} />
        ))}
      </div>
      {/* ...existing code... */}
    </div>
  );
}

export default App;
