# stock_prediction
The goal is to predict the stock movement using historical prices, social media, and correlations among stocks. 
In src folder you can find the model.py which has the functions needed for our model.
This file comprises three primary components:
1. Market Information Encoder (MIE) 
MIE encodes information from social media and stock prices to enhance market information quality, and outputs the market information input for VMD.

2. Variational Movement Decoder (VMD)
The purpose of VMD is to recurrently infer and decode the latent driven factor and the movement from the encoded market information.

3. Attentive Temporal Auxiliary (ATA) 
ATA integrates temporal loss through an attention mechanism for model training.

In src/main.py you can choose whether you want to train your model or you want to test it.

In src/datapipe.py you can load the dataset.

Train function and Test function has been implemented in src/executor.py.

src/stat_logger is where you can find the format of your output.

In src/config.yml you can set the parameters.

I also have uploaded the notebook that can run this project on Google Colab.
