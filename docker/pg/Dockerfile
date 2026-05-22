
# 基于 PostgreSQL 16 官方镜像，编译安装 pgvector + Apache AGE
# pg_trgm 是 PG 自带的 contrib 模块，无需额外安装

FROM postgres:16

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    postgresql-server-dev-16 \
    libreadline-dev \
    zlib1g-dev \
    flex \
    bison \
    && rm -rf /var/lib/apt/lists/*

# 关闭 git SSL 验证（解决内网/公司网络环境下 SSL 证书问题）
RUN git config --global http.sslVerify false

# ============ 安装 pgvector ============
RUN cd /tmp \
    && git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# ============ 安装 Apache AGE ============
RUN cd /tmp \
    && git clone --branch PG16/v1.5.0-rc0 https://github.com/apache/age.git \
    && cd age \
    && make \
    && make install \
    && rm -rf /tmp/age

# ============ 配置 shared_preload_libraries ============
# AGE 需要预加载共享库才能使用
RUN echo "shared_preload_libraries = 'age'" >> /usr/share/postgresql/postgresql.conf.sample

# 清理编译工具（可选，减小镜像体积）
RUN apt-get purge -y --auto-remove \
    build-essential \
    git \
    postgresql-server-dev-16 \
    libreadline-dev \
    zlib1g-dev \
    flex \
    bison \
    && rm -rf /var/lib/apt/lists/*
