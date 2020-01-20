const REDRAW_COUNTER = DATASET.length;
const DELAY = 1000;

let redrawIndex = 0;

let errorBandsX = [];
let errorBandsY = [];

let errorBandX = [];
let errorBandLY = [];
let errorBandUY = [];

let realX = [];
let realY = [];

let annX = [];
let annY = [];

let lstmX = [];
let lstmY = [];

const generateChart = () => {
    var errorBands = {
        x: errorBandsX,
        y: errorBandsY,
        fill: "tozerox",
        fillcolor: "rgba(0,100,80,0.2)",
        line: {
            color: "transparent"
        },
        name: "Error Bands",
        showlegend: false,
        type: "scatter"
    };

    var originalValues = {
        x: realX,
        y: realY,
        line: {
            color: "rgb(0,100,80)"
        },
        mode: "lines",
        name: "Original Value",
        type: "scatter"
    };

    var ann = {
        x: annX,
        y: annY,
        line: {
            color: "rgb(231,107,243)"
        },
        mode: "lines",
        name: "FBN",
        type: "scatter"
    };

    var lstm = {
        x: lstmX,
        y: lstmY,
        line: {
            color: "rgb(231,107,0)"
        },
        mode: "lines",
        name: "LSTM",
        type: "scatter"
    };

    var data = [errorBands, originalValues, ann, lstm];
    var layout = {
        paper_bgcolor: "rgb(255,255,255)",
        plot_bgcolor: "rgb(229,229,229)",
        xaxis: {
            gridcolor: "rgb(255,255,255)",
            showgrid: true,
            showline: false,
            showticklabels: true,
            tickcolor: "rgb(127,127,127)",
            ticks: "outside",
            zeroline: false
        },
        yaxis: {
            gridcolor: "rgb(255,255,255)",
            showgrid: true,
            showline: false,
            showticklabels: true,
            tickcolor: "rgb(127,127,127)",
            ticks: "outside",
            zeroline: false
        }
    };
    Plotly.newPlot("myDiv", data, layout, {
        showSendToCloud: true
    });
};

const refreshChart = () => {
    generateChart(); //generating an empty chart only for user to just see something at first not a blank page 

    const reDrawChart = setInterval(async() => {
        if (redrawIndex >= REDRAW_COUNTER) {
            clearInterval(reDrawChart); 
        } else {
            // call server

            const result = await axios.post("http://127.0.0.1:5000/predict", {
                gas_limit: DATASET[redrawIndex][0],
                gas_used: DATASET[redrawIndex][1],
                size: DATASET[redrawIndex][2],
                transaction_count: DATASET[redrawIndex][3],
                date: new Date()
            });

            const timeStamp = new Date(); // now

            //error boundary: if % we multiply if normal we use +- 
            // const ely = DATASET[redrawIndex][4] * 0.9;
            // const euy = DATASET[redrawIndex][4] * 1.1;
            const ely = DATASET[redrawIndex][4] - 0.1; 
            const euy = DATASET[redrawIndex][4] + 0.1;

            errorBandX.push(timeStamp);
            errorBandLY.push(ely);
            errorBandUY.push(euy);

            // console.log(errorBandX);
            // console.log(errorBandLY);
            // console.log(errorBandUY);

            for (let index = 0; index < errorBandX.length; index++) {
                errorBandsX.push(errorBandX[index]);
                errorBandsY.push(errorBandLY[index]);
            }

            for (let index = errorBandX.length - 1; index >= 0; index--) {
                errorBandsX.push(errorBandX[index]);
                errorBandsY.push(errorBandUY[index]);
            }

            // console.log("#######################################");
            // console.log(errorBandsX);
            // console.log(errorBandsY);

            realX.push(timeStamp);
            realY.push(DATASET[redrawIndex][4]);

            annX.push(timeStamp);
            // annY.push(Math.random());
            annY.push(result.data.ann_prediction[0]);

            lstmX.push(timeStamp);
            // lstmY.push(Math.random());
            lstmY.push(result.data.lstm_prediction[0]);

            generateChart();
            redrawIndex++;
        }
    }, DELAY);
};

refreshChart();