import tensorflow as tf
import gradio as gr
import numpy as np
import cv2
import os

classes = ["Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx"]
example_images = ["examples/" + f for f in os.listdir("examples")]

img_size = 400
model = tf.keras.models.load_model("CatClassifier.keras")

def model_predict(image):
    image = cv2.resize(image, (400,400))
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predictions = predictions[0]

    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]
    pred_dict = {}

    for i in range(len(classes)):
        pred_dict[classes[i]] = predictions[i]

    return predicted_class, pred_dict


def predict_breed(image):
    if image is None:
        return "Please attach an image first!", None
    
    return model_predict(image)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
          image_input = gr.Image(label="Cat Image")
          run_button = gr.Button(variant="primary")
          examples = gr.Examples(example_images,inputs=image_input)
        with gr.Column():
          breed_output = gr.Text(label="Predicted Breed", interactive=False)
          predict_labels = gr.Label(label="Class Probabilties")  
          
    run_button.click(fn=predict_breed, inputs=image_input, outputs=[breed_output, predict_labels])

if __name__ == "__main__":
    demo.launch()