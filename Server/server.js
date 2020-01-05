let  express = require('express');

let app = express();

app.use(function(req, res, next) {
    console.log(`${new Date()} - ${req.method} request for ${req.url}`);
    next();
});

app.use(express.static("../Client"));

app.listen(4000, function() {
    console.log("Serving static on 4000");
});


