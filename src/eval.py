def evaluate_model(model, dataloader, num_classes, device, csv_path, log_dir):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Acumula a confusion matrix
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast(device_type='cuda'):
                preds = torch.argmax(model(imgs), dim=1)

            pn = preds.cpu().numpy().flatten()
            mn = masks.cpu().numpy().flatten()
            valid = (mn >= 0) & (mn < num_classes)
            cm += np.bincount(num_classes * mn[valid] + pn[valid], minlength=num_classes**2).reshape(num_classes, num_classes)

    # ----- Métricas globais -----
    total = cm.sum()
    pix_acc = np.trace(cm) / total if total > 0 else 0.0

    # IoU por classe
    ious = []
    for i in range(num_classes):
        TP = cm[i,i]
        FP = cm[:,i].sum() - TP
        FN = cm[i,:].sum() - TP
        denom = TP + FP + FN
        ious.append(TP/denom if denom > 0 else np.nan)
    miou = np.nanmean(ious)

    # Kappa de Cohen
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    expected = np.outer(row_sum, col_sum) / total if total > 0 else np.zeros_like(cm)
    kappa = (np.trace(cm) - np.trace(expected)) / (total - np.trace(expected)) if total - np.trace(expected) != 0 else 0

    # Dice por classe
    dice = []
    for i in range(num_classes):
        TP = cm[i,i]
        FP = cm[:,i].sum() - TP
        FN = cm[i,:].sum() - TP
        denom = 2*TP + FP + FN
        dice.append(2*TP/denom if denom > 0 else np.nan)

    # Weighted IoU
    w = cm.sum(axis=1) / total
    wiou = np.nansum(np.array(ious) * w)

    # Métricas de precisão/recall/F1
    metrics = {}
    for i in range(num_classes):
        TP = cm[i,i]
        FP = cm[:,i].sum() - TP
        FN = cm[i,:].sum() - TP
        precision = TP/(TP+FP) if TP+FP > 0 else 0
        recall    = TP/(TP+FN) if TP+FN > 0 else 0
        f1        = 2*precision*recall/(precision+recall) if precision+recall > 0 else 0
        metrics[i] = {'precision': precision, 'recall': recall, 'f1-score': f1, 'support': TP+FN}

    # Salva CSV
    df = pd.DataFrame(metrics).T
    df['Pixel Accuracy'] = pix_acc
    df['Mean IoU']       = miou
    df['Kappa']          = kappa
    df['Dice']           = dice
    df['Weighted IoU']   = wiou
    df.to_csv(csv_path)

    # Matriz de confusão
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(log_dir,'confusion_matrix.png'))
    plt.close()

    logging.info(f'Pixel Acc: {pix_acc:.4f} | mIoU: {miou:.4f} | Kappa: {kappa:.4f}')
    return pix_acc, miou, kappa, dice, wiou

