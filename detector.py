import joblib
import sys
import argparse

def load_model(model_path='phishing_model.joblib'):
    try:
        artifacts = joblib.load(model_path)
        return artifacts['model'], artifacts['vectorizer']
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found")
        sys.exit(1)

def predict_email(text, model, vectorizer,
                  phishing_thresh=0.75, short_limit=20, phishing_tokens=None):
    if phishing_tokens is None:
        phishing_tokens = {"free","win","bonus","urgent","claim","offer","prince"}

    if len(text) < short_limit and not any(tok in text.lower() for tok in phishing_tokens):
        return "LEGIT", 0.9

    vec    = vectorizer.transform([text])
    probs  = model.predict_proba(vec)[0]
    p_phishing = probs[list(model.classes_).index(1)]

    if p_phishing >= phishing_thresh:
        return "LEGIT", p_phishing
    else:
        return "phishing", 1 - p_phishing

def main():
    parser = argparse.ArgumentParser(description='Email phishing Detector CLI')
    parser.add_argument('email', nargs='?', type=str, help='Email text to analyze (enclose in quotes)')
    parser.add_argument('--file', type=str, help='Path to text file containing email')
    args = parser.parse_args()

    model, vectorizer = load_model()

    if args.file:
        with open(args.file, 'r') as f:
            text = f.read()
    elif args.email:
        text = args.email
    else:
        print("\nPaste the email content below. Press Enter twice to finish:\n")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == '':
                    break
                lines.append(line)
            except EOFError:
                break
        text = "\n".join(lines)

    result, confidence = predict_email(text, model, vectorizer)

    print(f"\n{'='*30}")
    print(f"Result: {'🚨 PHISHING' if result == 'phishing' else '✅ LEGIT'}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"{'='*30}\n")

if __name__ == '__main__':
    main()

