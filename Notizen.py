# Hier werden Preprocessing-Schritte ausgeführt
# s ist hier dann Differnzbild (Pixelraum_Fourier)


# Start Preprocessing: RGB_clean, RGB_filmed (könnte man irgendwo implementieren - normales Python)
# Input: Fourier_clean und Fourier_filmed  (Datatype: tf.complex)
## fft2(RGB_clean), fft2(RGB_filmed)
# Erster Schritt:  Fourier_clean - Fourier_filmed (im komplexen Raum? Bisher im Pixel_Fourier)

# HIER STARTET MODEL (UNet)
# c1(Differenzbild)
# y_pred: Segmentierungsmaske (weights anpassen) U-Net (als Blackbox verwenden!)
# Output: leicht verändertes Differenzbild (Werte zwischen 0..1 -> Sigmoid)
# HIER ENDET MODEL (UNet)

# -> Alpha Blending "soft" (alternativ die Funktionen aus Facilities verwenden)(reine Berechnung)
# Input: Differenzbild UND zwei Variablen (tf.Placeholder) für Fourier_clean und Fourier_filmed
# -> result = ifft (reine Berechnung)
# Input: Fourier_processed Output: Bild in RGB-Domain

# ALIASING-CNN verwenden
# predict_aliasing(result) (nicht die weights verändern!)
# Input: RGB/Bild    Output: 2D Matrix mit 0..1 Wahrscheinlichkeiten für Aliasing

# loss funktion
# Loss wird pro Kachel berechnet
# Fake Fake_lables: ausschließlich kein Aliasing! -> 2D Matrix mit 0
## y_pred = Kacheln
# crossentropy -> loss_fn(Fake_lables, y_pred)


# HIGH-LEVEL Programmierung
# rgb_processed = alpha_blending(u_net_output, rgb_clean_fft, rgb_filmed_fft)
# y_pred = model_2(rgb_processed, training=False)
# loss = loss_fn(y_pred, fake_labels)
# gradients = tape.gradient(loss, model.trainable_weights)
# optimizer.apply_gradients(zip(gradients, model.trainable_weights))
