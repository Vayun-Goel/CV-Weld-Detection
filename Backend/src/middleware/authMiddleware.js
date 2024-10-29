const jwt = require("jsonwebtoken");

const verifyToken = (req,res,next) => {
    let token;
    let authHeader = req.headers.Authorization || req.headers.authorization;
    if(authHeader && authHeader.startsWith("Bearer")){
        token = authHeader.split(" ")[1];

        if(!token){
            res.status(401).json({message: "authorization denied, no token"});
        }

        try{
            const decode = jwt.verify(token,process.env.JWT_SECRET);
            req.user = decode;
            next();
        }
        catch{
            res.status(401).json({message:"invalid token"});
        }
    }
}

module.exports = verifyToken;