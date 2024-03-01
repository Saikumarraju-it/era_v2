# Back Propagation Steps:

## Step1: Get the forword equations , sigmoids and the loss function.
![image](https://github.com/Saikumarraju-it/era_v2/assets/66442795/70aea608-2cf4-40ea-a877-e1ee53606ec2)

## Step2: Start the back propagation by taking the changes to the total loss wrt w5 and w6. First start from the bottom , if we change w5 then it will change h1, if we change h1 then a_h1 will change, if we change a_h1 then total loss will change.

<img width="376" alt="Screenshot 2024-03-02 at 12 06 20 AM" src="https://github.com/Saikumarraju-it/era_v2/assets/66442795/0554d708-3d2d-4d47-9e74-13270606d627">

## Step3: Similary calculate for the hidden layer weights(w5,w6,w7,w8)
<img width="416" alt="Screenshot 2024-03-02 at 12 06 43 AM" src="https://github.com/Saikumarraju-it/era_v2/assets/66442795/75fb8130-d2b4-4b82-947b-5ed607bc9562">

## Step4: Calculate the gradients for a_h1 and a_h2, using chain rule and the intermediate outputs(from last layer)(from node to to node)
<img width="587" alt="Screenshot 2024-03-02 at 12 07 16 AM" src="https://github.com/Saikumarraju-it/era_v2/assets/66442795/b5f6bd37-e165-499e-834f-37ec5dab49be">

## Step5: Calculate the gradines for w1,w2,w3 using from and to nodes, chain rules
<img width="401" alt="Screenshot 2024-03-02 at 12 07 45 AM" src="https://github.com/Saikumarraju-it/era_v2/assets/66442795/2f8e6082-1804-4175-a705-7b21c4a63f83">

## Step6: solve the partial derivations using below formulas.
<img width="855" alt="Screenshot 2024-03-02 at 12 08 18 AM" src="https://github.com/Saikumarraju-it/era_v2/assets/66442795/5937884b-fbfe-4e14-9b84-97ef0168c41f">

## Step7: calculate loss and all gradients using the targets t1(0.5) and t2(0.5) , inputs i1 and i2 , initial random weights and learinig parameter (for some epochs)
<img width="626" alt="Screenshot 2024-03-02 at 12 35 51 AM" src="https://github.com/Saikumarraju-it/era_v2/assets/66442795/cd86a617-15d9-47b0-b8b6-2881585e02ef">


# Error Graph

## when learning rate = 0.1
<img width="494" alt="Screenshot 2024-03-02 at 12 36 58 AM" src="https://github.com/Saikumarraju-it/era_v2/assets/66442795/3e5b8710-08c4-4ba5-b331-cb70b94a31e7">
## when learning rate = 0.2
<img width="494" alt="Screenshot 2024-03-02 at 12 37 56 AM" src="https://github.com/Saikumarraju-it/era_v2/assets/66442795/3cd596d7-855b-4753-be47-469a05c5ddb8">

## when learning rate = 0.5
<img width="482" alt="Screenshot 2024-03-02 at 12 38 23 AM" src="https://github.com/Saikumarraju-it/era_v2/assets/66442795/68a10b9b-f85f-4434-923f-75494d316e95">

## when learning rate = 0.8
<img width="499" alt="Screenshot 2024-03-02 at 12 38 50 AM" src="https://github.com/Saikumarraju-it/era_v2/assets/66442795/f594c977-ac5b-43d3-93ea-175e73a086a2">

## when learning rate = 1.0
<img width="481" alt="Screenshot 2024-03-02 at 12 39 10 AM" src="https://github.com/Saikumarraju-it/era_v2/assets/66442795/6b35925b-9fe6-484a-b89f-b0e95a3f1c02">

## when learning rate = 2.0
<img width="487" alt="Screenshot 2024-03-02 at 12 39 30 AM" src="https://github.com/Saikumarraju-it/era_v2/assets/66442795/a79daf6a-26af-490f-b737-25c044d2513a">



