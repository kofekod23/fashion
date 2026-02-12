# AI Fashion Search

Moteur de recherche visuel pour la mode, propulsé par [Fashion CLIP](https://huggingface.co/patrickjohncyh/fashion-clip) et [Weaviate](https://weaviate.io/).

Recherchez des vêtements par texte ou par image grâce à la recherche hybride (dense vectors + BM25).

## Fonctionnalités

- **Recherche par texte** — décrivez un vêtement en langage naturel ("black leather jacket", "robe d'été fleurie")
- **Recherche par image** — uploadez une photo pour trouver des articles similaires
- **Recherche intelligente** — parsing sémantique des requêtes avec filtres automatiques (couleur, catégorie, prix, genre, marque)
- **Chatbot guidé** — sélection du genre → contexte de style → recherche affinée
- **Panel admin** — gestion des exemples, FAQ, paramètres du site, logs avec géolocalisation
- **Fiche produit** — vue détaillée avec toutes les photos d'un produit

## Tech Stack

| Composant | Technologie |
|-----------|-------------|
| Embeddings | [Fashion CLIP](https://huggingface.co/patrickjohncyh/fashion-clip) (512d) |
| Vector DB | Weaviate 1.28 |
| Backend | FastAPI + Uvicorn |
| Dataset | [ASOS e-commerce](https://huggingface.co/datasets/UniqueData/asos-e-commerce-dataset) |
| Infra | GCP (VM + GPU NVIDIA), Docker Compose |
| Runtime | Python 3.11 |

## Démarrage rapide

### Prérequis

- Docker & Docker Compose
- ~4 Go de RAM pour Weaviate
- GPU NVIDIA + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (pour le mode GPU)

### 1. Configuration

```bash
cp .env.example .env
```

### 2. Mode CPU (local / VM sans GPU)

```bash
# Lancer Weaviate + app
docker compose up -d

# Indexer (500 produits par défaut)
docker compose run --rm indexer
```

### 3. Mode GPU (GCP avec NVIDIA)

```bash
# Lancer Weaviate + app GPU
docker compose --profile gpu up -d

# Indexer avec GPU (2000 produits par défaut)
docker compose --profile setup-gpu run --rm indexer-gpu
```

Pour ajuster le nombre de produits :

```bash
docker compose --profile setup-gpu run --rm indexer-gpu sh -c "python scripts/index_asos.py --max-items 5000 --max-images-per-product 3"
```

L'application est disponible sur [http://localhost:8000](http://localhost:8000).

## Développement local (sans Docker)

```bash
# Installer les dépendances
pip install -e ".[dev]"

# Lancer Weaviate seul
docker compose up -d weaviate

# Indexer
python scripts/index_asos.py --max-items 100

# Lancer le serveur
uvicorn src.web.app:app --reload --port 8000
```

## API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Interface web |
| `POST` | `/search` | Recherche texte (JSON: `query`, `gender`, `context`) |
| `POST` | `/search/image` | Recherche par image (multipart) |
| `GET` | `/featured` | Produits aléatoires pour la homepage |
| `GET` | `/product/{id}` | Détail d'un produit |
| `GET` | `/stats` | Statistiques de l'index |
| `GET` | `/health` | Health check |

## Structure du projet

```
├── src/
│   ├── config.py              # Configuration (env vars)
│   ├── models/
│   │   ├── base.py            # Interface modèle
│   │   └── clip_model.py      # Fashion CLIP wrapper
│   ├── database/
│   │   └── weaviate_client.py # Client Weaviate
│   ├── search/
│   │   ├── engine.py          # Moteur de recherche hybride
│   │   └── query_parser.py    # Parsing sémantique des requêtes
│   └── web/
│       ├── app.py             # Application FastAPI
│       ├── routes.py          # Routes API + admin
│       └── templates/
│           └── index.html     # Interface frontend
├── scripts/
│   └── index_asos.py         # Pipeline d'ingestion ASOS
├── tests/
├── docker-compose.yml
├── Dockerfile              # Image CPU
├── Dockerfile.gpu          # Image GPU (CUDA 12.1)
└── pyproject.toml
```

## Licence

Projet personnel de Julien G.
