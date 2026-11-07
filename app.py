import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).parent

FEATURE_LABELS = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Красное вино'
}


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def get_test_probabilities(_model, X):
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def get_batch_probabilities(_model, X):
    return _model.predict_proba(X)[:, 1]


def read_batch_csv(content, feature_names):
    try:
        data = pd.read_csv(
            io.BytesIO(content),
            sep=None,
            engine='python',
            encoding='utf-8-sig'
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError,
            UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            'Не удалось прочитать CSV. Сохраните файл в UTF-8 '
            'с разделителем «запятая» или «точка с запятой».'
        ) from error

    missing = [name for name in feature_names if name not in data.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))

    if data.empty:
        raise ValueError('В CSV нет строк с данными.')

    if 'target' in data.columns:
        raise ValueError(
            'Удалите столбец target: для предсказания нужны только признаки вин.'
        )

    features = data.loc[:, feature_names].apply(pd.to_numeric, errors='coerce')
    invalid = features.columns[
        ~np.isfinite(features.to_numpy(dtype=float)).all(axis=0)
    ].tolist()

    if invalid:
        raise ValueError(
            'Проверьте числовые значения и пропуски в столбцах: '
            + ', '.join(invalid)
        )

    if not features['is_red'].isin([0, 1]).all():
        raise ValueError('В столбце is_red допустимы только 0 и 1.')

    return data, features


st.set_page_config(
    page_title='Оценка качества вина',
    page_icon='🍷',
    layout='wide'
)

st.title('🍷 Оценка качества вина')
st.caption('Хорошее вино: экспертная оценка не ниже 7 из 10.')

try:
    model = load_model()
    test_data = load_test_data()
    config = json.loads(
        (APP_DIR / 'config.json').read_text(encoding='utf-8')
    )
    feature_names = list(model.feature_names_in_)
    X_test = test_data.loc[:, feature_names]
    y_test = test_data['target']
    default_threshold = float(config['threshold'])
    if not 0 <= default_threshold <= 1:
        raise ValueError('Порог в config.json должен быть от 0 до 1.')
    probabilities = get_test_probabilities(model, X_test)
except (OSError, ValueError, KeyError, AttributeError, ImportError) as error:
    st.error(f'Не удалось загрузить модель или данные: {error}')
    st.stop()

st.sidebar.header('Настройки подборки')
threshold = st.sidebar.slider(
    'Порог классификации',
    min_value=0.0,
    max_value=1.0,
    value=default_threshold,
    step=0.001,
    format='%.3f',
    key='threshold'
)
st.sidebar.caption(f'Точное значение порога: {threshold:.6f}')
st.sidebar.write(
    'Вино попадает в подборку, если вероятность хорошего качества '
    'не меньше выбранного порога.'
)

tab_single, tab_metrics, tab_batch, tab_weights = st.tabs([
    'Оценка одного вина',
    'Модельные метрики',
    'Пакетное предсказание',
    'Что выучила модель'
])

with tab_single:
    st.subheader('Введите характеристики вина')
    st.caption(
        'Диапазоны полей взяты из тестовых данных. '
        'Начальные числовые значения — медианы.'
    )

    wine_type = st.radio(
        'Тип вина',
        ['Белое', 'Красное'],
        index=int(X_test['is_red'].mode().iloc[0]),
        horizontal=True,
        key='wine_type'
    )

    wine_values = {'is_red': int(wine_type == 'Красное')}
    numeric_features = [name for name in feature_names if name != 'is_red']
    input_columns = st.columns(3)

    for index, name in enumerate(numeric_features):
        minimum = float(X_test[name].min())
        maximum = float(X_test[name].max())
        median = float(X_test[name].median())
        step = 0.01
        number_format = '%.2f'

        if name == 'density':
            step = 0.0001
            number_format = '%.5f'
        elif name == 'chlorides':
            step = 0.001
            number_format = '%.4f'
        elif name in ['free_sulfur_dioxide', 'total_sulfur_dioxide']:
            step = 1.0

        with input_columns[index % 3]:
            wine_values[name] = st.number_input(
                FEATURE_LABELS[name],
                min_value=minimum,
                max_value=maximum,
                value=median,
                step=step,
                format=number_format,
                help=f'Столбец CSV: {name}',
                key=f'input_{name}'
            )

    single_wine = pd.DataFrame([wine_values], columns=feature_names)
    single_probability = float(model.predict_proba(single_wine)[0, 1])

    st.metric('Вероятность хорошего вина', f'{single_probability:.1%}')
    if single_probability >= threshold:
        st.success('Решение модели: рекомендовать в подборку.')
    else:
        st.info('Решение модели: не включать в подборку.')

    st.caption(f'Решение принято при пороге {threshold:.3f}.')

with tab_metrics:
    st.subheader('Качество на тестовой выборке')
    predictions = (probabilities >= threshold).astype(int)
    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)

    metric_columns = st.columns(3)
    metric_columns[0].metric('Precision', f'{precision:.3f}')
    metric_columns[1].metric('Recall', f'{recall:.3f}')
    metric_columns[2].metric('F1', f'{f1:.3f}')

    st.caption(
        f'В подборку выбрано {int(predictions.sum())} '
        f'из {len(y_test)} тестовых вин.'
    )

    confusion = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()
    false_positive_rate = fp / (fp + tn)

    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=['Остальные', 'Хорошие']
    ).plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title('Матрица ошибок')
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, probabilities)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(
        y_test, probabilities
    )
    roc_auc = metrics.roc_auc_score(y_test, probabilities)
    average_precision = metrics.average_precision_score(y_test, probabilities)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Случайная модель')
    axes[0].scatter(
        false_positive_rate, recall,
        color='darkorange', s=80, zorder=5,
        label=f'Порог {threshold:.3f}'
    )
    axes[0].set_title('ROC-кривая')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')

    axes[1].step(
        recall_curve, precision_curve, where='post',
        label=f'Average Precision = {average_precision:.3f}'
    )
    axes[1].axhline(
        y_test.mean(), linestyle='--', color='gray', label='Случайная модель'
    )
    axes[1].scatter(
        recall, precision,
        color='darkorange', s=80, zorder=5,
        label=f'Порог {threshold:.3f}'
    )
    axes[1].set_title('PR-кривая')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best')

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        'Если повысить порог, отбор станет строже и вин в подборке будет меньше. '
        'Если понизить порог, в подборку попадёт больше вин: можно найти больше '
        'хороших, но возрастает риск рекомендовать посредственные.'
    )
    st.caption(
        'FP — посредственное вино попало в подборку. '
        'FN — хорошее вино было пропущено. '
        'Порог меняет отмеченные точки, сами ROC- и PR-кривые остаются прежними.'
    )

with tab_batch:
    st.subheader('Предсказания для CSV-файла')
    st.write(
        'Загрузите таблицу с 12 признаками, без столбца target. '
        'В is_red укажите 1 для красного вина и 0 для белого. '
        'Порядок столбцов может быть любым.'
    )
    st.caption('Нужные столбцы: ' + ', '.join(feature_names))
    st.download_button(
        'Скачать пример CSV',
        data=X_test.head(5).to_csv(index=False).encode('utf-8-sig'),
        file_name='wine_example.csv',
        mime='text/csv',
        key='download_example'
    )

    uploaded_file = st.file_uploader('CSV с признаками вин', type=['csv'])
    st.caption(f'Для всех строк используется общий порог {threshold:.3f}.')

    if uploaded_file is not None:
        try:
            batch_data, batch_features = read_batch_csv(
                uploaded_file.getvalue(), feature_names
            )
            batch_probabilities = get_batch_probabilities(model, batch_features)
            result = batch_data.copy()
            result['probability_good'] = batch_probabilities
            result['prediction'] = (batch_probabilities >= threshold).astype(int)
            result['decision'] = np.where(
                result['prediction'] == 1,
                'Рекомендовать',
                'Не включать'
            )

            st.success(f'Обработано вин: {len(result)}')
            st.dataframe(result)
            st.download_button(
                'Скачать предсказания',
                data=result.to_csv(index=False).encode('utf-8-sig'),
                file_name='wine_predictions.csv',
                mime='text/csv',
                key='download_predictions'
            )
        except ValueError as error:
            st.error(str(error))

with tab_weights:
    st.subheader('Коэффициенты логистической регрессии')
    coefficients = pd.Series(
        model[-1].coef_[0], index=feature_names, name='Коэффициент'
    ).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [FEATURE_LABELS[name] for name in coefficients.index],
        coefficients.to_numpy(),
        color=np.where(coefficients.to_numpy() >= 0, '#2a7f62', '#b34a4a')
    )
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Коэффициент после масштабирования признаков')
    ax.set_title('Вклад признаков в оценку модели')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    strongest_positive = coefficients[coefficients > 0].nlargest(3)
    strongest_negative = coefficients[coefficients < 0].nsmallest(3)

    st.write(
        '**Сильнее всего за хорошее вино:** '
        + ', '.join(
            f'{FEATURE_LABELS[name]} ({value:+.3f})'
            for name, value in strongest_positive.items()
        )
    )
    st.write(
        '**Сильнее всего против:** '
        + ', '.join(
            f'{FEATURE_LABELS[name]} ({value:+.3f})'
            for name, value in strongest_negative.items()
        )
    )
    st.caption(
        'При увеличении признака положительный коэффициент повышает оценку '
        'хорошего качества, отрицательный понижает, если остальные признаки '
        'фиксированы. Признаки масштабированы: изменение на одну единицу '
        'соответствует одному стандартному отклонению тренировочных данных. '
        'Это зависимости, найденные моделью, а не доказательство причинности.'
    )
