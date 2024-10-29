const User = require("../models/userModel");
const bcript = require("bcryptjs");
const jwt = require("jsonwebtoken");

const register = async (req,res) => {
    try{
        const {username,password,role} = req.body;
        const hashedpassword = await bcript.hash(password,10);
        
        const newUser = new User({username,password: hashedpassword, role});
        await newUser.save();
        res.status(201).json({message: `User registered: ${username} with role ${role}`});
    }
    catch{
        res.status(500).json({message: `user could not be registered`});
    }
};

const login = async(req,res) => {
    try{
        const {username,password} = req.body;

        const user = await User.findOne({username});

        if(!user){
            res.status(404).json({message: `User not found : ${username}`});
        }

        const match = await bcript.compare(password,user.password);

        if(!match){
            res.status(400).json({message: `invalid credentials`});
        }

        const token = jwt.sign({id: user._id, role: user.role}, process.env.JWT_SECRET, {expiresIn: "1h"});

        res.status(200).json({token});
    }
    catch{
        res.status(500).json({message: `user could not be logged in`});
    }
};

module.exports = {register,login};