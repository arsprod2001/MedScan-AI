import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, layer_name="conv2d_2"):
        """Initialise Grad-CAM avec le modèle et la couche cible"""
        self.model = model
        self.layer_name = layer_name
        self.grad_model = self._create_grad_model()

    def _create_grad_model(self):
        """Crée un modèle qui retourne les activations et les prédictions"""
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(self.layer_name).output, self.model.output]
        )
        return grad_model

    def compute_heatmap(self, image, class_idx=None):
        """
        Calcule la carte d'activation Grad-CAM
        Args:
            image: Image d'entrée (batch de 1 image)
            class_idx: Index de la classe cible (None pour utiliser la classe prédite)
        Returns:
            heatmap: Carte d'activation normalisée
        """
        # Conversion en tenseur
        img_tensor = tf.convert_to_tensor(image)
        
        # Enregistrement des gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        
        # Calcul des gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Pooling global des gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiplication des poids par les activations
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalisation entre 0 et 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, original_img, alpha=0.4):
        """
        Superpose la heatmap sur l'image originale
        Args:
            heatmap: Carte d'activation Grad-CAM
            original_img: Image originale (format OpenCV BGR)
            alpha: Transparence de la heatmap
        Returns:
            superimposed_img: Image avec heatmap superposée
        """
        # Redimensionnement de la heatmap
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        
        # Application d'une colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superposition avec l'image originale
        superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
        
        return superimposed_img

    def visualize(self, image_path, model_input_size=(224, 224), save_path=None):
        """
        Pipeline complet de visualisation Grad-CAM
        Args:
            image_path: Chemin vers l'image
            model_input_size: Taille d'entrée du modèle
            save_path: Chemin pour sauvegarder le résultat
        Returns:
            Tuple: (image originale, heatmap, image superposée)
        """
        # Chargement et prétraitement de l'image
        original_img = cv2.imread(image_path)
        img = cv2.resize(original_img, model_input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Prédiction et calcul Grad-CAM
        preds = self.model.predict(img)
        class_idx = np.argmax(preds[0])
        heatmap = self.compute_heatmap(img, class_idx)
        
        # Superposition
        superimposed_img = self.overlay_heatmap(heatmap, original_img)
        
        # Conversion pour affichage
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        # Sauvegarde si nécessaire
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        
        return original_img, heatmap, superimposed_img

def plot_gradcam_results(original_img, heatmap, superimposed_img, class_name, confidence):
    """Visualisation des résultats Grad-CAM avec matplotlib"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Image Originale")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Carte d'Activation")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(f"Grad-CAM: {class_name} ({confidence:.1%})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Charger un modele pré-entraîné
    model = tf.keras.models.load_model("medical_cnn.h5")
    
    # Initialiser Grad-CAM
    gradcam = GradCAM(model)
    
    # Exécuter sur une image exemple
    img_path = "example_medical_image.jpg"
    original_img, heatmap, superimposed_img = gradcam.visualize(img_path, save_path="gradcam_output.jpg")
    
    # Afficher les résultats (supposons classe et confiance pour l'exemple)
    plot_gradcam_results(original_img, heatmap, superimposed_img, "Pneumonia", 0.92)
