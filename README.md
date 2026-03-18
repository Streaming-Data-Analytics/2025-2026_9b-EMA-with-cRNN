# 2025/2026: Accelerating Convergence in Streaming Time Series with Weight EMA for cLSTM and cGRU

Optional project of the [Streaming Data Analytics](https://emanueledellavalle.org/teaching/streaming-data-analytics-2025-26/) course provided by Politecnico di Milano.

Student: **[To be assigned]**
_____
# Brief Description
This project proposes an original extension of the concepts introduced in the 2023 paper *"Improving Online Continual Learning Performance and Stability with Temporal Ensembles"*. 

While the original paper focuses on mitigating the stability gap in convolutional architectures for image data, this project explores a different hypothesis tailored for **streaming data with explicit temporal dependencies**. 

Your objective is to implement **Continuous LSTM (cLSTM)** and **Continuous GRU (cGRU)** backbones, and investigate whether maintaining an Exponential Moving Average (EMA) of the model weights helps these recurrent architectures **converge faster** to new temporal patterns immediately after a concept drift. You will evaluate these models on the **Weather** and **AirQuality** streaming datasets.

______

# Background
Data streams often exhibit **temporal dependence**, where current outcomes are highly correlated with past observations. To model this, architectures like **cLSTM** and **cGRU** accumulate incoming data points into mini-batches and extract temporal sequences using a sliding window.

When learning continuously from an evolving stream, these models must constantly adapt to **concept drifts**. Typically, adjusting the recurrent weights to learn a entirely new temporal pattern takes several training steps, resulting in a prolonged period of poor predictive performance before the model finally converges to the new concept.

Taking inspiration from Soutif-Cormerais et al., we maintain a "target" model by applying an **Exponential Moving Average (EMA)** to the recurrent model's "online" weights:
$$\theta_{EMA}^{(t)} = \lambda \cdot \theta_{EMA}^{(t-1)} + (1 - \lambda) \cdot \theta_{online}^{(t)}$$

**Our Hypothesis:** We hypothesize that using these EMA weights for inference (and potentially as a regularization signal) smooths the optimization trajectory. This temporal ensembling can act as a stabilizing momentum, filtering out noisy gradient updates and allowing the cLSTM and cGRU to **accelerate their convergence**, reaching high predictive performance in fewer mini-batches after a drift occurs.

# Project Goals
1. **Implement cLSTM and cGRU:** Set up the continuous recurrent baselines capable of processing incoming data streams via a sliding window mechanism.
2. **Integrate Weight EMA:** Implement the target model and the EMA weight update for the recurrent architectures. 
3. **Analyze Convergence Speed:** Evaluate the models on non-stationary time series, specifically focusing on the adaptation phase after each concept drift to measure which model converges to the new optimal representation faster.

### Hyperparameters
To ensure consistency and comparability, you must strictly use the following hyperparameters for your experiments:
* **Epochs number ($E$)**: `10`
* **Mini-batch size ($B$)**: `128`
* **Learning rate**: `0.01`
* **LSTM hidden layer size ($H_{LSTM}$)**: `50`
* **GRU hidden size ($H_{GRU}$)**: `25`
* **Window Size ($W$)**: `10` for AirQuality, `11` for Weather.

# Datasets
You will evaluate your implementation on two streaming datasets characterized by temporal dependence and engineered abrupt concept drifts:
- [**Weather**](https://polimi365-my.sharepoint.com/:x:/g/personal/10780444_polimi_it/IQDCgaKx14mVR7zozlZM0yStAaR0ONNRc-_GHGV1z75_Jyg?e=mXeuBp)
- [**AirQuality**](https://polimi365-my.sharepoint.com/:x:/g/personal/10780444_polimi_it/IQC1J1X1g-HvR506YGFuJqvpAZ-xFPmN58wkuZg909pOk3U?e=gczUQi)
The **complete description** is available [here](https://polimi365-my.sharepoint.com/:b:/g/personal/10780444_polimi_it/IQAZ8E8JZMobRpkBEq91srZqATZrL9RkyQN0jeUVscWcD1A?e=HX9tjE). 

They are associated with the weather and air pollution domains. Features are already standardized. 8 different binary classification labels are engineered by comparing current values (of a hidden feature, which is not given) to previous median or minimum values within a specific temporal window. Each classification function corresponds to a specific concept. For instance, one concept may assign 1 if the current avalue has increased since the previous timestamp, and 0 otherwise. Another task may assign 1 if the current value is greater than the mean of the previous 10, … and so on. 

Concept drifts in this case are **real** and **abrupt**.

The column **task** represents the index of the concept. You can use it to trigger a cPNN expansion (when it changes, it indicates a drift and you must call the method `add_new_column`).
The **Target** column represents the label associated with each data point. 

# Evaluation Metrics
You will use the **Test-then-Train (Prequential)** evaluation approach:
* The model receives a new feature vector $X_t$ and outputs a prediction $\hat{y}_t$ (using the EMA weights when considering the EMA version).
* The prediction is scored against the true label $y_t$.
* The model accumulates the data and trains periodically (updating the online weights via backpropagation and the EMA weights via moving average).

You must track the **Rolling Cohen's Kappa** over time. **Crucially**, the rolling window must be reset immediately after each concept drift. This reset allows you to isolate and clearly observe the steepness of the recovery curve (the convergence speed) without the influence of the previous concept's accuracy.

# Deliverables
1. **Code Implementation:** A PyTorch-based repository containing the cLSTM and cGRU implementations with the integrated Weight EMA logic.
2. **Jupyter Notebook:** A comprehensive presentation of the results. 
   - Plot the Rolling Cohen's Kappa over the data stream.
   - Zoom in on the moments immediately following the concept drifts to compare the slopes of the recovery curves between standard cLSTM/cGRU and their EMA-enhanced counterparts.
   - Discuss your findings: Does temporal ensembling effectively accelerate convergence on new temporal patterns?

## Note for Students
* Clone the created repository offline;
* Add your name and surname into the Readme file;
* Add a `requirements.txt` file for code reproducibility;
* Commit your changes to your local repository and push them online.
