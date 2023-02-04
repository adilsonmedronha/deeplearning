This is an implementation to verify the behavior of the following phenomenon: 

> *"When the decoder is **linear** and $L$ is the mean squared error, an undercomplete **autoencoder** learns to span the **same subspace as PCA**. In this case, an autoencoder trained to perform the copying task has learned the principal subspace of the training data as a side effect."*  
Goodfellow, I., Bengio, Y., & Courville, A. 2016. Deep learning, (p. 494)

I have implemented both linear and non-linear autoencoders, as well as a PCA. \
You  can find here a visualization of the latent spaces and a comparison of the reconstruction quality of each model.



# PCA vs NLAE vs LAE 

## Latent Space
<img src="assets\nlae_latent_space.png"  width="500" height="450">
<img src="assets\lae_latent_space.png"  width="500" height="450">
<img src="assets\pca_latent_space.png"  width="500" height="450">


## Reconstructions Overview
<img src="assets\reconstructions.png"  width="800" height="800">