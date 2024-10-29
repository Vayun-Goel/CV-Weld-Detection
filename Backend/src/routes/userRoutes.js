const express = require("express");
const verifyToken = require("../middleware/authMiddleware");
const authorizedRoles = require("../middleware/roleMiddleware");
const router = express.Router();

//user
router.get('/hi',(req,res)=>{
    res.json('welcome user');
})

module.exports = router;