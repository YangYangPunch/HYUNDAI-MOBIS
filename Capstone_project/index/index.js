const AWS = require('aws-sdk');
const dynamo = new AWS.DynamoDB({
    accessKeyId : "AKIA4UOGJ4VZFKJ7JLUI",
    secretAccessKey : "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",
    region : 'ap-northeast-2'
})

var params = {
    TableName : 'ValveOnOff'
}


// sensordata
exports.handler = function(event, context, callback) {
    console.log("Received event: ", event);
    dynamo.scan(params, (err, data) => {
        var data = {
            "greetings": JSON.stringify(data.Items[0].id.S) + " , " + JSON.stringify(data.Items[0].control.S)
        };
        callback(null, data);
    });
};

