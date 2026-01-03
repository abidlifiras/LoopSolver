"""
main.py
- run the pipeline once (mine patterns, train model)
- then start the Flask app
"""
from src.utils import full_pipeline
from src.app import app
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = full_pipeline()
    print("Pipeline terminé. Patterns trouvés:")
    for p in res["patterns"]:
        print(p)
    print("\nRègles (decision tree):\n")
    print(res["rules"])
    print("\nDémarrage de l'interface web sur http://127.0.0.1:5000")
    app.run(debug=True)
