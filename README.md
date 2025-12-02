# 🎨 ChromaFlow

*A little app that turns your workouts into art.*

ChromaFlow is a fun, visual way to look at your movement.
Instead of charts or stats, every workout you log becomes a piece of generative artwork — shaped by the type of exercise, how long you moved, the intensity, and even a bit of your profile (like age and gender).

It’s movement → color → creativity.

---

## ✨ What You Can Do

* Add workouts (running, yoga, cycling, walking, weightlifting)
* Choose duration + intensity
* Add a bit of personal info so the art feels more *you*
* Watch your latest workout turn into a unique visual design
* Save your creations in a built-in gallery
* Download artwork as a PNG

---

## 🧪 What’s Behind the Curtain

ChromaFlow uses:

* **Streamlit** for the web interface
* **Matplotlib** + **NumPy** to generate the artwork
* A small system of color/shape rules that react to your workout details

Every time you log something, the app creates a new generative “motif” — floral, wave-like, or bursting — depending on the workout type and intensity.

---

## ▶️ Run It Yourself

If you want to test it locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deploying

This app is built to run easily on **Streamlit Community Cloud**, so you can share it with others through a simple web link.

---

## 💡 Why I Made This

ChromaFlow reimagines fitness data in a more creative, human way.
It’s a reminder that movement doesn’t have to be measured only in numbers —
it can also be something expressive, personal, and visually rewarding.
