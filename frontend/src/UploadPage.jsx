import React, { useState } from 'react';
import './UploadPage.css';

const UploadPage = () => {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      console.log("File uploaded:", selectedFile.name);
    }
  };

  return (
    <div className="upload-container">
      <header className="upload-header">
        <h1>Phygital</h1>
        <p>AI-Powered Spatial Analytics</p>
      </header>

      <main className="upload-card">
        <div className="upload-icon">🗺️</div>
        <h2>Upload Floor Plan</h2>
        <p>Select a JPEG, PNG, or PDF of your store layout to begin tracking.</p>
        
        <label className="file-label">
          <input 
            type="file" 
            onChange={handleFileChange} 
            accept="image/*,.pdf" 
            hidden 
          />
          {file ? `Selected: ${file.name}` : "Browse Files"}
        </label>

        {file && (
          <button className="continue-btn" onClick={() => alert("Moving to Scanner...")}>
            Process Layout
          </button>
        )}
      </main>
    </div>
  );
};

export default UploadPage;