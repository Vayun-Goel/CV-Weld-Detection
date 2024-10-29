const express = require("express");
const {register,login} = require("../controllers/authController");
const verifyToken = require("../middleware/authMiddleware");
const authorizedRoles = require("../middleware/roleMiddleware");
const router = express.Router();

router.post('/register',verifyToken, authorizedRoles("admin"),register);
router.post('/login',login);

module.exports = router;
