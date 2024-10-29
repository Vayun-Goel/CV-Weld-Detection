const express = require('express');
const path = require('path');
const fs = require('fs');
const dotenv = require("dotenv").config();
const dbConnect = require("./config/dbConnect");
const authRoutes = require("./routes/authRoutes")
const userRoutes = require("./routes/userRoutes")
const uploadRoute = require("./routes/uploadRoute")
const adminRoutes = require("./routes/adminRoutes")
const verifyToken = require("./middleware/authMiddleware");
const authorizedRoles = require("./middleware/roleMiddleware");


const app = express();
dbConnect();

//middleware
app.use(express.json());
app.use(express.static('public'));
app.use('/uploads', express.static('uploads'));

//routes
app.use("/api/auth",authRoutes);
app.use("/api/admin",verifyToken, authorizedRoles("admin"),adminRoutes);
app.use("/api/user",verifyToken, authorizedRoles("admin","user"),userRoutes);
app.use("upload",uploadRoute);

// Create input and output directories if they don't exist
const inputDir = path.join(__dirname, 'uploads', 'input');
const outputDir = path.join(__dirname, 'uploads', 'output');
fs.mkdirSync(inputDir, { recursive: true });
fs.mkdirSync(outputDir, { recursive: true });

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Express server running on port ${PORT}`);
});