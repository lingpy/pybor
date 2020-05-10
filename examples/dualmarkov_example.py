def validate_loan_detection_dual_basis(data, method='kni', smoothing=0.5,
                                       order=3, trainfrac=0.8):

    train_data, test_data = train_test_split(data, test_size=1-trainfrac)

    dual_model = DualMarkov(train_data, method=method,
                            order=order, smoothing=smoothing)

    print("Evaluate train dataset.")
    predictions = dual_model.predict_data(train_data)
    train_metrics = evaluate.evaluate_model(predictions, train_data)
    evaluate.print_evaluation(train_metrics)

    print("Evaluate test dataset.")
    predictions = dual_model.predict_data(test_data)
    test_metrics = evaluate.evaluate_model(predictions, test_data)
    evaluate.print_evaluation(test_metrics)

    return dual_model, test_metrics


# =============================================================================
#
#  Validate loan detection - native model effectiveness.
#
# =============================================================================
def validate_loan_detection_native_basis(data, method='kni', smoothing=0.5,
                                    order=3, p=0.995, trainfrac=0.8):

    train_data, test_data = train_test_split(data, test_size=1-trainfrac)

    native_model = NativeMarkov(train_data, method=method,
                                order=order, smoothing=smoothing, p=p)

    print("Evaluate train dataset.")
    predictions = native_model.predict_data(train_data)
    train_metrics = evaluate.evaluate_model(predictions, train_data)
    evaluate.print_evaluation(train_metrics)

    print("Evaluate test dataset.")
    predictions = native_model.predict_data(test_data)
    test_metrics = evaluate.evaluate_model(predictions, test_data)
    evaluate.print_evaluation(test_metrics)

    return native_model, test_metrics


