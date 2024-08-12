from sklearn.preprocessing import StandardScaler
import pandas as pd
import nn

if __name__ == '__main__':
    file_path = 'data/train.csv'
    train_df = pd.read_csv(file_path)
    img_dir = 'path/to/your/image_folder'

    def preprocess_data(data):
        sorted_indices = sorted(range(len(data)), key=lambda i: str(int(data[i][0].item())))
        return sorted_indices

    scaler = StandardScaler()
    evaluation_df = pd.read_csv('data/test.csv')
    ed = evaluation_df.iloc[:, :].values

    X_train = train_df.iloc[:, :-6].values
    order = preprocess_data(X_train)
    X_train = scaler.fit_transform(X_train)
    X_train = X_train[order]
    test_order = preprocess_data(ed)
    ed = ed[test_order]

    y4_train = train_df.iloc[:, -6].values
    y4_train = scaler.fit_transform(y4_train.reshape(-1, 1))
    y4_train = y4_train[order]
    nn.FNN(X_train, y4_train, "model4")
    prediction4 = nn.predict(ed, "model4").detach().cpu().numpy().reshape(-1, 1)
    prediction4 = scaler.inverse_transform(prediction4).reshape(-1)

    y11_train = train_df.iloc[:, -5].values
    y11_train = scaler.fit_transform(y11_train.reshape(-1, 1))
    y11_train = y11_train[order]
    nn.FNN(X_train,y11_train,"model11")
    prediction11 = nn.predict(ed,"model11").detach().cpu().numpy().reshape(-1, 1)
    prediction11 = scaler.inverse_transform(prediction11).reshape(-1)

    y18_train = train_df.iloc[:, -4].values
    y18_train = scaler.fit_transform(y18_train.reshape(-1, 1))
    y18_train = y18_train[order]
    nn.FNN(X_train,y18_train,"model18")
    prediction18 = nn.predict(ed,"model18").detach().cpu().numpy().reshape(-1, 1)
    prediction18 = scaler.inverse_transform(prediction18).reshape(-1)
    
    y26_train = train_df.iloc[:, -3].values
    y26_train = scaler.fit_transform(y26_train.reshape(-1, 1))
    y26_train = y26_train[order]
    nn.FNN(X_train,y26_train,"model26")
    prediction26 = nn.predict(ed,"model26").detach().cpu().numpy().reshape(-1, 1)
    prediction26 = scaler.inverse_transform(prediction26).reshape(-1)
    
    y50_train = train_df.iloc[:, -2].values
    y50_train = scaler.fit_transform(y50_train.reshape(-1, 1))
    y50_train = y50_train[order]
    nn.FNN(X_train,y50_train,"model50")
    prediction50 = nn.predict(ed,"model50").detach().cpu().numpy().reshape(-1, 1)
    prediction50 = scaler.inverse_transform(prediction50).reshape(-1)
    
    y3112_train = train_df.iloc[:, -1].values
    y3112_train = scaler.fit_transform(y3112_train.reshape(-1, 1))
    y3112_train = y3112_train[order]
    nn.FNN(X_train,y3112_train,"model3112")
    prediction3112 = nn.predict(ed,"model3112").detach().cpu().numpy().reshape(-1, 1)
    prediction3112 = scaler.inverse_transform(prediction3112).reshape(-1)

    id = evaluation_df.iloc[:, 0]
    id = id[test_order]
    data = {
        'id': id,
        'X4': prediction4,
        'X11': prediction11,
        'X18': prediction18,
        'X26': prediction26,
        'X50': prediction50,
        'X3112': prediction3112
    }

    df = pd.DataFrame(data)

    df.to_csv('21019206_lu.csv', index=False)
