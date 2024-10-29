const authorizedRoles = (...allowedRoles) => {
    return (req,res,next) => {
        console.log(`${req.user.role} has invoked a function`);
        if(!allowedRoles.includes(req.user.role)){
            return res.status(403).json("access denied");
        }
        next();
    }
}

module.exports = authorizedRoles;