"""
tests/test_ingestion.py

Validação da pipeline de ingestão de documentos:
  - Existência dos artefatos processados (chunks + parquet)
  - Estrutura e integridade dos chunks gerados
  - Metadados obrigatórios em cada chunk
  - Consistência do corpus (textos não vazios, módulos reconhecidos)
  - Presença do índice vetorial (BM25 + Qdrant)

Não faz chamadas ao LLM nem ao Qdrant — testes determinísticos sobre artefatos em disco.
"""
import json
import pickle
from pathlib import Path

# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
VECTORSTORE_DIR = ROOT_DIR / "data" / "vectorstore"

MODULOS_ESPERADOS = {
    "Módulo 1", "Módulo 2", "Módulo 3", "Módulo 4", "Módulo 5",
    "Módulo 6", "Módulo 7", "Módulo 8", "Módulo 9", "Módulo 10",
    "Módulo 11", "REN 1000/2021",
}
# ---------------------------------------------------------------------------


class TestIngestaoArtefatos:
    """Valida existência e estrutura dos artefatos de ingestão."""

    def test_parquet_indicadores_existe(self):
        """O Parquet de indicadores operacionais deve existir em data/processed."""
        parquet = PROCESSED_DIR / "aneel_indicadores.parquet"
        assert parquet.exists(), f"Parquet não encontrado: {parquet}"
        assert parquet.stat().st_size > 0, "Parquet está vazio"

    def test_chunks_json_existem(self):
        """Deve haver pelo menos um arquivo *_chunks.json em data/processed."""
        chunks_files = sorted(PROCESSED_DIR.glob("*_chunks.json"))
        assert len(chunks_files) > 0, (
            f"Nenhum arquivo *_chunks.json encontrado em {PROCESSED_DIR}. "
            "Execute a pipeline de ingestão antes dos testes."
        )

    def test_bm25_index_existe(self):
        """O índice BM25 (bm25.pkl) deve existir em data/vectorstore."""
        bm25_path = VECTORSTORE_DIR / "bm25.pkl"
        assert bm25_path.exists(), f"Índice BM25 não encontrado: {bm25_path}"
        assert bm25_path.stat().st_size > 0, "bm25.pkl está vazio"

    def test_qdrant_vectorstore_existe(self):
        """O diretório Qdrant deve existir em data/vectorstore."""
        assert VECTORSTORE_DIR.exists(), f"Diretório vectorstore não encontrado: {VECTORSTORE_DIR}"
        # Pelo menos algum arquivo de dados do Qdrant
        qdrant_files = list(VECTORSTORE_DIR.iterdir())
        assert len(qdrant_files) > 0, "Diretório vectorstore está vazio"


class TestIntegridadeChunks:
    """Valida estrutura interna e qualidade dos chunks gerados."""

    def _carregar_todos_chunks(self) -> list[dict]:
        """Carrega todos os chunks de todos os arquivos *_chunks.json."""
        todos = []
        for arquivo in sorted(PROCESSED_DIR.glob("*_chunks.json")):
            with open(arquivo, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            todos.extend(chunks)
        return todos

    def test_chunks_nao_vazios(self):
        """Todos os chunks devem ter texto não-vazio."""
        todos = self._carregar_todos_chunks()
        assert len(todos) > 0, "Nenhum chunk encontrado"

        vazios = [i for i, c in enumerate(todos) if not c.get("texto", "").strip()]
        assert len(vazios) == 0, f"{len(vazios)} chunks com texto vazio. Índices: {vazios[:5]}"

    def test_chunks_tem_metadados_obrigatorios(self):
        """Cada chunk deve ter os metadados: 'modulo', 'pagina', 'documentos'."""
        todos = self._carregar_todos_chunks()
        campos_obrigatorios = {"modulo", "pagina", "documentos"}
        faltando = []

        for i, chunk in enumerate(todos):
            meta = chunk.get("metadados", {})
            ausentes = campos_obrigatorios - set(meta.keys())
            if ausentes:
                faltando.append({"idx": i, "ausentes": list(ausentes)})

        assert len(faltando) == 0, (
            f"{len(faltando)} chunks sem metadados obrigatórios. "
            f"Primeiros casos: {faltando[:3]}"
        )

    def test_chunks_modulos_reconhecidos(self):
        """Os módulos dos chunks devem estar no conjunto esperado do PRODIST."""
        todos = self._carregar_todos_chunks()
        modulos_no_corpus = set(c["metadados"]["modulo"] for c in todos)
        desconhecidos = modulos_no_corpus - MODULOS_ESPERADOS

        assert len(desconhecidos) == 0, (
            f"Módulos não reconhecidos no corpus: {desconhecidos}. "
            "Verifique a lógica de extrair_modulo() em chunking.py"
        )

    def test_chunks_tamanho_razoavel(self):
        """Chunks devem ter entre 50 e 2000 caracteres (detecta chunking corrompido)."""
        todos = self._carregar_todos_chunks()
        muito_curtos = [i for i, c in enumerate(todos) if len(c.get("texto", "")) < 50]
        muito_longos = [i for i, c in enumerate(todos) if len(c.get("texto", "")) > 2000]

        pct_curtos = len(muito_curtos) / len(todos) if todos else 0
        pct_longos = len(muito_longos) / len(todos) if todos else 0

        # Tolerância: até 1% de chunks fora do range é aceitável (bordas de documento)
        assert pct_curtos <= 0.01, (
            f"{len(muito_curtos)} chunks ({pct_curtos:.1%}) com menos de 50 chars"
        )
        assert pct_longos <= 0.01, (
            f"{len(muito_longos)} chunks ({pct_longos:.1%}) com mais de 2000 chars"
        )

    def test_chunks_paginas_validas(self):
        """Todos os chunks devem ter número de página inteiro positivo."""
        todos = self._carregar_todos_chunks()
        invalidos = [
            i for i, c in enumerate(todos)
            if not isinstance(c.get("metadados", {}).get("pagina"), int)
            or c["metadados"]["pagina"] < 1
        ]
        assert len(invalidos) == 0, (
            f"{len(invalidos)} chunks com página inválida. Primeiros: {invalidos[:5]}"
        )

    def test_total_chunks_suficiente(self):
        """O corpus deve ter pelo menos 500 chunks (ingestão mínima razoável)."""
        todos = self._carregar_todos_chunks()
        assert len(todos) >= 500, (
            f"Corpus tem apenas {len(todos)} chunks. "
            "Esperado pelo menos 500 para cobertura adequada do PRODIST."
        )

    def test_bm25_index_carregavel(self):
        """O índice BM25 deve ser carregável via pickle e ter vocabulário não-vazio."""
        bm25_path = VECTORSTORE_DIR / "bm25.pkl"
        if not bm25_path.exists():
            return  # já coberto por test_bm25_index_existe

        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)

        # BM25Okapi tem o atributo idf ou corpus_size
        assert hasattr(bm25, "get_scores"), "Objeto BM25 não tem método get_scores()"
        # Verifica que o corpus foi indexado (corpus_size > 0)
        corpus_size = getattr(bm25, "corpus_size", None)
        if corpus_size is not None:
            assert corpus_size > 0, "BM25 indexado com corpus vazio"

    def test_corpus_cobre_todos_modulos(self):
        """O corpus deve ter chunks de todos os 12 módulos esperados."""
        todos = self._carregar_todos_chunks()
        modulos_presentes = set(c["metadados"]["modulo"] for c in todos)
        ausentes = MODULOS_ESPERADOS - modulos_presentes
        assert len(ausentes) == 0, (
            f"Módulos sem nenhum chunk no corpus: {ausentes}. "
            "Verifique se todos os PDFs foram ingeridos."
        )