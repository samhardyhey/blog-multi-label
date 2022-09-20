from otso_dictionary_wrapper.model import DictionaryClassifier

if __name__ == "__main__":
    from clear_bow.classifier import DictionaryClassifier

    dictionaries = {
        "customer_service": ["customer service", "service", "experience"],
        "pricing": ["expensive", "cheap", "dear", "dollars", "cents"],
        "billing": ["quarterly", "online", "phone"],
        "product": [
            "quality",
            "product",
            "superior",
            "inferior",
            "fast",
            "efficient",
            "range",
            "selection",
            "replaced",
        ],
        "competitor": ["another provider", "competitor", "leaving", "will not return"],
    }
    dc = DictionaryClassifier(
        classifier_type="multi_label", label_dictionary=dictionaries
    )
