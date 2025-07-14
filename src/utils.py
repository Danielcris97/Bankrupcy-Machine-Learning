import joblib
from tqdm import tqdm
from contextlib import contextmanager

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar
    
    Example Usage:
    
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from src.utils import tqdm_joblib

    with tqdm_joblib(tqdm(desc="My Parallel Task", total=10)):
        Parallel(n_jobs=2)(delayed(some_function)(i) for i in range(10))
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback

# Para probar este módulo directamente si se ejecuta
if __name__ == "__main__":
    print("Este es un módulo de utilidades.")
    print("Contiene funciones auxiliares para el proyecto de ML.")
    print("\nDemostración de tqdm_joblib con Parallel:")

    from joblib import Parallel, delayed
    import time

    def example_task(i):
        time.sleep(0.1) # Simular algún trabajo
        return i * i

    try:
        with tqdm_joblib(tqdm(desc="Calculando cuadrados", total=10)):
            results = Parallel(n_jobs=-1)(delayed(example_task)(i) for i in range(10))
        print("\nEjemplo completado. Resultados:", results)
    except Exception as e:
        print(f"\nNo se pudo ejecutar la demostración de tqdm_joblib (asegúrate de que 'joblib' y 'tqdm' estén instalados): {e}")