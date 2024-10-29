const express = require("express");
const verifyToken = require("../middleware/authMiddleware");
const authorizedRoles = require("../middleware/roleMiddleware");
const router = express.Router();

//admin routes
router.get('/hi',(req,res)=>{
    res.json('welcome admin');
})

module.exports = router;