// // App.js
// import React, { useState } from 'react';
// import axios from 'axios';
// import './App.css';

// function App() {
//   const [inputImage, setInputImage] = useState(null);
//   const [outputImage, setOutputImage] = useState(null);
//   const [defect, setDefect] = useState([]);
//   const [isLoading, setIsLoading] = useState(false);
//   const [fileName, setFileName] = useState('');

//   const handleFileUpload = async (event) => {
//     const file = event.target.files[0];
//     if (file) {
//       setFileName(file.name);
//       setInputImage(null); 
//       setOutputImage(null); 
//       const formData = new FormData();
//       formData.append('image', file);

//       setIsLoading(true);

//       try {
//         const response = await axios.post('/upload', formData, {
//           headers: { 'Content-Type': 'multipart/form-data' }
//         });

//         setInputImage(response.data.inputImage);
//         setOutputImage(response.data.outputImage);
//         // console.log(response)
//         setDefect(response.data.detectedClasses); 
//       } catch (error) {
//         console.error('Error uploading file:', error);
//         setDefect('Error processing image');
//       } finally {
//         setIsLoading(false);
//       }
//     }
//   };

//   return (
//     <div className="App">
//       <header className="App-header">
//         <h1>Image Processing Website</h1>
//       </header>
//       <main className="App-main">
//         <section className="upload-section">
//           <h2>Upload Image</h2>
//           <div className="file-input-wrapper">
//             <button className="file-input-button">Choose File</button>
//             <input
//               type="file"
//               onChange={handleFileUpload}
//               accept="image/*"
//               className="file-input"
//             />
//             <span className="file-name">{fileName || 'No file chosen'}</span>
//           </div>
//         </section>
//         <div className="image-container">
//           <section className="image-section">
//             <h2>Input Image</h2>
//             {inputImage && <img src={inputImage} alt="Input" className="processed-image" />}
//           </section>
//           <section className="image-section">
//             <h2>Output Image</h2>
//             {outputImage && <img src={outputImage} alt="Output" className="processed-image" />}
//           </section>
//         </div>
//         <section className="defect-section">
//           <h2>Detected Defect</h2>
//           <div style={{
//             display: 'flex',
//             justifyContent: 'center',
//             alignItems: 'center',
//             gap: "10px"
//           }}>          
//             {defect.map((entity) => {
//             return(
//               <p>{entity}</p>
//             )
//           })}
//           </div>

//         </section>
//         {isLoading && <div className="loading">Processing image...</div>}
//       </main>
//     </div>
//   );
// }

// export default App;


import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [inputImage, setInputImage] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [defects, setDefects] = useState([]); // Changed name to defects and initialized as empty array
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState('');
  const [error, setError] = useState(null);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileName(file.name);
      setInputImage(null);
      setOutputImage(null);
      setDefects([]); // Clear previous defects
      setError(null); // Clear previous errors
      
      const formData = new FormData();
      formData.append('image', file);

      setIsLoading(true);

      try {
        const response = await axios.post('/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });

        setInputImage(response.data.inputImage);
        setOutputImage(response.data.outputImage);
        
        // Ensure detectedClasses is always an array
        const detectedClasses = Array.isArray(response.data.detectedClasses) 
          ? response.data.detectedClasses 
          : [response.data.detectedClasses];
        
        setDefects(detectedClasses);
      } catch (error) {
        console.error('Error uploading file:', error);
        setError('Error processing image');
        setDefects([]); // Clear defects on error
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Image Processing Website</h1>
      </header>
      <main className="App-main">
        <section className="upload-section">
          <h2>Upload Image</h2>
          <div className="file-input-wrapper">
            <button className="file-input-button">Choose File</button>
            <input
              type="file"
              onChange={handleFileUpload}
              accept="image/*"
              className="file-input"
            />
            <span className="file-name">{fileName || 'No file chosen'}</span>
          </div>
        </section>
        <div className="image-container">
          <section className="image-section">
            <h2>Input Image</h2>
            {inputImage && <img src={inputImage} alt="Input" className="processed-image" />}
          </section>
          <section className="image-section">
            <h2>Output Image</h2>
            {outputImage && <img src={outputImage} alt="Output" className="processed-image" />}
          </section>
        </div>
        <section className="defect-section">
          <h2>Detected Defects</h2>
          {error ? (
            <p className="error">{error}</p>
          ) : (
            <div className="defects-container" style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              gap: "10px"
            }}>
              {defects.map((defect, index) => (
                <p key={index}>{defect}</p>
              ))}
            </div>
          )}
        </section>
        {isLoading && <div className="loading">Processing image...</div>}
      </main>
    </div>
  );
}

export default App;