import React, { Component } from "react";
import axios from "axios";

import "./chart.css";
import "./homeP.css";

import RTChart from "react-rt-chart";

import DATA from "./../../DATA";
const DELAY = 1000;

class Home extends Component {
  state = { dataSize: 0, counter: 0, data: null };

  componentDidMount = () => {
    this.setState({ dataSize: DATA.length });

    const chartData = setInterval(async () => {
      if (this.state.counter >= this.state.dataSize) {
        clearInterval(chartData);
      } else {
        const result = await axios.post("http://127.0.0.1:5000/predict", {
          gas_limit: DATA[this.state.counter][0],
          gas_used: DATA[this.state.counter][1],
          size: DATA[this.state.counter][2],
          transaction_count: DATA[this.state.counter][3],
          date: new Date()
        });
        // console.log(result.data);
        let data = {
          date: new Date(),
          FBN_Prediction: result.data.ann_prediction[0],
          LSTM_Prediction: result.data.lstm_prediction[0],
          Original_Value: DATA[this.state.counter][4]
        };
        const index = this.state.counter + 1;
        this.setState({ counter: index, data });
      }
    }, DELAY);
  };

  render() {
    return (
      <div className="main-container">
        <div className="home-rt-chart rounded border border-primary">
          <RTChart
            style={{ height: "50vh" }}
            fields={["FBN_Prediction", "LSTM_Prediction", "Original_Value"]}
            data={this.state.data}
          />
        </div>
      </div>
    );
  }
}

export default Home;
