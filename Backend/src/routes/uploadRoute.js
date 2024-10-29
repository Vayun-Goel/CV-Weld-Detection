const express = require("express");
const path = require('path');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');
const router = express.Router();

// Configure multer for file upload
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
      cb(null, inputDir)
    },
    filename: function (req, file, cb) {
      cb(null, Date.now() + '-' + file.originalname)
    }
  });
  
const upload = multer({ storage: storage });

router.post('/', upload.single('image'), async (req, res) => {
    if (!req.file) {
      return res.status(400).send('No file uploaded.');
    }
  
    const inputPath = req.file.path;
    const outputFilename = 'processed_' + req.file.filename;
    const outputPath = path.join(outputDir, outputFilename);
  
    const formData = new FormData();
    formData.append('file', fs.createReadStream(inputPath));
  
    try {
      const flaskResponse = await axios.post('http://localhost:5000/process', formData, {
        headers: formData.getHeaders(),
      });
  
      // Write the base64 encoded image to a file
      const base64Data = flaskResponse.data.processed_image.replace(/^data:image\/png;base64,/, "");
      fs.writeFileSync(outputPath, base64Data, 'base64');
      console.log(flaskResponse.data.detected_classes)
      res.json({
        inputImage: '/uploads/input/' + req.file.filename,
        outputImage: '/uploads/output/' + outputFilename,
        detectedClasses: flaskResponse.data.detected_classes
      });
    } catch (error) {
      console.error('Error processing image:', error);
      res.status(500).send('Error processing image');
    }
});

module.exports = router;