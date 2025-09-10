from dataclasses import dataclass, field
from typing import Optional, Any
from util.json_df import auto_convert_numeric, flatten_columns

import re
import os
import json
import difflib
import warnings
from decouple import config

from datetime import timedelta
from itertools import combinations

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import MessagesState, StateGraph, END

from langchain.indexes import SQLRecordManager, index
from langchain_postgres.vectorstores import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.schema import Document
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import traceable
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate)
from langchain_core.prompts.prompt import PromptTemplate
from regressChain import getBestModel
from geoLift import powerAnalysis
from rich.console import Console
from rich.markdown import Markdown
from allGraphs import all_graphs
from rpy2.robjects import pandas2ri
from allStats import cleaningData, edaAnalysis
from graphAnalysis import run_test
from cleanData import full_data_cleaning
from modified_reg import *
from causalModel import causalModel
from synthetiControl import synthetic_control
from langdetect import detect
from fill_na_ts import SmartImputer
from expand_df import DailyDistributionExpander
from time_series_analysis import TimeSeriesAnalyzer
from abTestAnalysis import get_analysis
from assoc_rules import run_assoc
warnings.simplefilter("ignore")


@dataclass
class Herakleitos:
    df_raw: pd.DataFrame
    chat_model: Any
    user_id: str
    memory: Optional[Any] = None
    eda_cleaned_df: Optional[pd.DataFrame] = None
    causal_cleaned_df: Optional[pd.DataFrame] = None
    ts_cleaned_df: Optional[pd.DataFrame] = None
    console: Console = field(default_factory=Console)
    checkpointer: Any = field(default_factory=InMemorySaver)
    def __post_init__(self):
        """Clean and stage raw data on class instantiation.

            Prepares datasets for downstream EDA, causal inference, and time-series analysis
            (e.g., type coercions, missing-value handling, feature filtering).

            Side Effects:
                Populates cleaned data attributes (e.g., ``self.eda_cleaned_df``, ``self.ts_cleaned_df``,
                ``self.causal_cleaned_df``).
            """
        pandas2ri.activate()

        # ---------- EDA Cleaning ----------
        if self.eda_cleaned_df is None:
            print("🔹 Cleaning Data For EDA:")
            df_eda_target = self.df_raw.copy()

            dict_or_list_cols = [
                col for col in df_eda_target.columns
                if df_eda_target[col].apply(lambda x: isinstance(x, (dict, list))).any()
            ]

            safe_df = df_eda_target.drop(columns=dict_or_list_cols)
            #convert numeric
            safe_df = auto_convert_numeric(safe_df)
            eda_cleaned = cleaningData(safe_df)
            eda_cleaned = pandas2ri.rpy2py_dataframe(eda_cleaned)

            dict_data = df_eda_target[dict_or_list_cols]
            flattened = flatten_columns(dict_data, drop_original=False)

            self.eda_cleaned_df = pd.concat([
                eda_cleaned.reset_index(drop=True),
                flattened.reset_index(drop=True)
            ], axis=1)

            print("✅ EDA data is converted...")

        # ---------- Causal Cleaning ----------
        if self.causal_cleaned_df is None:
            print("🔹 Cleaning Data For Causal:")
            df_causal_target = self.df_raw.copy()

            dict_or_list_cols = [
                col for col in df_causal_target.columns
                if df_causal_target[col].apply(lambda x: isinstance(x, (dict, list))).any()
            ]

            safe_df = df_causal_target.drop(columns=dict_or_list_cols)
            safe_df = auto_convert_numeric(safe_df)

            causal_cleaned,*_ = full_data_cleaning(safe_df)

            dict_data = df_causal_target[dict_or_list_cols]
            flattened = flatten_columns(dict_data, drop_original=False)

            self.causal_cleaned_df = pd.concat([
                causal_cleaned.reset_index(drop=True),
                flattened.reset_index(drop=True)
            ], axis=1)

            print("✅ Causal data is converted...")

        if self.ts_cleaned_df is None:
            print("🔹 Cleaning Data For Time Series:")
            df_ts_target = self.df_raw.copy()

            dict_or_list_cols = [
                col for col in df_ts_target.columns
                if df_ts_target[col].apply(lambda x: isinstance(x, (dict, list))).any()
            ]

            ts_safe_df = df_ts_target.drop(columns=dict_or_list_cols)
            ts_safe_df = auto_convert_numeric(safe_df)
            imputer = SmartImputer(strategy="auto")
            num_cols = ts_safe_df.select_dtypes(include=[np.number]).columns.tolist()
            for col in num_cols:
                ts_safe_df = imputer.impute_campaign_summary(ts_safe_df, target_column=col)


            self.ts_cleaned_df = ts_safe_df.copy()

            print("✅ Time Series data is converted...")

        # ---------- df_raw Flatten ----------
        self.df_raw = flatten_columns(self.df_raw)

        # ---------- Memory ----------
        if self.memory:
            print("💾 Saving cleaned data to memory...")
            self._save_context_to_memory("eda")
            self._save_context_to_memory("causal")
            self._save_context_to_memory("ts")


    def _save_context_to_memory(self, query_type: str):
        """Persist cleaned data context in memory for agent pipelines.

            Args:
                query_type: One of ``{"eda", "causal", "ts"}`` to select which cleaned view
                    should be cached and reused by agents.

            Side Effects:
                Updates in-memory stores/checkpointer for subsequent agent calls.
            """
        if query_type == "eda":
            df = self.eda_cleaned_df
        elif query_type == "causal":
            df = self.causal_cleaned_df
        else:
            df = self.ts_cleaned_df
        df_copy = df.copy()

        for col in df_copy.select_dtypes(include=["datetime64"]).columns:
            df_copy[col] = df_copy[col].astype(str)

        self.memory.save_context(
            {"input": f"{query_type.capitalize()} Data"},
            {"output": json.dumps({"data": df_copy.to_dict(orient='list'), "date_col": "date"})}
        )

    @staticmethod
    def load_docs_from_file(path: str, source_name: str) -> list[Document]:
        """Load a document from disk and prepare it for vector indexing.

            Args:
                path: Absolute or relative path to the source document.
                source_name: Logical source identifier to tag the document with.

            Returns:
                A normalized document object or list of chunks, ready for embedding/indexing.

            Raises:
                FileNotFoundError: If ``path`` does not exist.
                ValueError: If file type is unsupported.
            """
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return [Document(page_content=content, metadata={"source": source_name})]

    @staticmethod
    def init_pgvector_retriever(docs, collection_name: str, namespace: str, embeddings_model):
        """Initialize a PgVector-backed retriever.

            Args:
                docs: Iterable of documents/chunks to index.
                collection_name: PgVector collection/table name.
                namespace: Logical namespace to isolate indexes.
                embeddings_model: Embedding model instance or identifier.

            Returns:
                A configured retriever object usable by RAG agents.
            """
        connection = config("PGVECTOR_CONNECTION_STRING")

        vectorstore = PGVector(
            embeddings=embeddings_model,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )

        record_manager = SQLRecordManager(
            namespace,
            db_url=connection,
        )
        record_manager.create_schema()

        index(
            docs,
            record_manager,
            vectorstore,
            cleanup="incremental",
            source_id_key="source"
        )

        return vectorstore.as_retriever(search_kwargs={"k": 5})

    @staticmethod
    def init_pg_rag_retriever(
            agent_name: str,
            lang: str,
            file_path: str = None,
            embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Build a complete RAG pipeline (load → embed → index → retrieve).

            Args:
                agent_name: Identifier for the consumer agent.
                lang: Language code used for prompt templates (e.g., ``"tr"``, ``"en"``).
                file_path: Path of the document to ingest.
                embedding_model_name: Name or handle of the embedding model.

            Returns:
                A ready-to-use retriever bound to the ingested corpus.
            """
        source = f"{agent_name}_{lang}"
        collection = f"{source}_col"
        namespace = f"{source}_ns"

        if file_path is None:
            file_path = f"services/AI/RAG/texts/{source}.txt"

        docs = Herakleitos.load_docs_from_file(file_path, source_name=source)

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        retriever = Herakleitos.init_pgvector_retriever(
            docs=docs,
            collection_name=collection,
            namespace=namespace,
            embeddings_model=embeddings
        )

        return retriever


    def generate_prompt(self, query, prompt_tr, prompt_en, lang="tr", **kwargs):
        """Create a chat prompt template from the user query and language.

            Args:
                query: End-user query in natural language.
                prompt_tr: Turkish system template string.
                prompt_en: English system template string.
                lang: Language selector (``"tr"`` or ``"en"``).
                **kwargs: Extra template variables.

            Returns:
                A ``ChatPromptTemplate`` (or equivalent) ready for the LLM.
            """

        # 📌 1. Sistem rolü her zaman aynı
        system_prompt = SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant who answers clearly and with context."
        )

        # 📌 2. Kullanıcı mesajı şablonu seçilir
        human_prompt_template = prompt_tr if lang == "tr" else prompt_en
        human_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)

        # 📌 3. ChatPromptTemplate oluşturulur
        chat_prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            human_prompt
        ])

        # 📌 4. input dictionary’yi hazırla
        prompt_inputs = {"query": query, **kwargs}

        return chat_prompt, prompt_inputs

    async def analyze_single_column(self, column, query):
        """Run deep analysis for a single column using an agent + RAG context.

            Args:
                column: Target column name in the dataset.
                query: Natural language question to steer the analysis.

            Yields:
                Text or Markdown chunks describing insights, tests, and plots.
            """
        df = self.eda_cleaned_df.copy()
        time_cols = [col for col in df.columns if "date" in col.lower() or "start" in col.lower()]
        if time_cols:
            date_col = time_cols[0]
            try:
                df = df.sort_values(by=date_col)
                df = df.set_index(date_col)
            except Exception as e:
                print(f"⚠️ Can't convert date column: {e}")

        # Agent oluştur
        python_tool = PythonREPLTool()
        create_agent = create_pandas_dataframe_agent(
            llm=self.chat_model,
            df=df,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            verbose=True,
            tools=[python_tool],
            callbacks=[StreamingStdOutCallbackHandler()]
        )

        # Agent çalıştır
        try:
            response = await create_agent.ainvoke(query)
            agent_analysis = response.get("output", "") if isinstance(response, dict) else str(response)
            if not agent_analysis.strip():
                yield "⚠️ Agent returned empty analysis."
                return
        except Exception as e:
            yield f"⚠️ Agent error: {e}"
            return

        # Dil tespiti
        try:
            lang = detect(query)
        except:
            lang = "en"

        # RAG
        retriever = Herakleitos.init_pg_rag_retriever(agent_name="analyze_agent", lang=lang)
        try:
            documents = retriever.get_relevant_documents(query)
            rag_context = "\n".join([doc.page_content for doc in documents])
        except Exception as e:
            rag_context = ""
            print(f"⚠️ Failed to fetch RAG context: {e}")

        # Promptlar
        prompt_en = """
        🎯 **Your Role:** You are a data scientist with access to two inputs:

        📊 **Agent Analysis (focused on column '{column}'):**
        {agent_analysis}

        📚 **Contextual Knowledge from Domain:**
        {rag_context}

        ❓ Based on the above, explain the situation related to '{column}' and the user query: '{query}'.
        """.strip()

        prompt_tr = """
        🧠 **Rolün:** Bir veri bilimci olarak iki girdiye sahipsiniz:

        📊 **Veriden Elde Edilen Agent Analizi ('{column}' kolonu üzerine):**
        {agent_analysis}

        📚 **Alan Bilgisinden Gelen Bağlamsal Bilgi:**
        {rag_context}

        ❓ Yukarıdaki bilgilere göre '{column}' kolonu ve '{query}' hakkında durumu açıklayın.
        """.strip()

        chat_prompt, inputs = self.generate_prompt(
            query=query,
            prompt_tr=prompt_tr,
            prompt_en=prompt_en,
            lang=lang,
            column=column,
            agent_analysis=agent_analysis,
            rag_context=rag_context
        )

        chain = chat_prompt | self.chat_model

        async for chunk in chain.astream(inputs):
            yield chunk.content if hasattr(chunk, "content") else str(chunk)

    async def analyze_agent(self, query):
        """Execute general exploratory analysis with a LangChain-style agent.

            Args:
                query: Free-form exploratory question across multiple columns/metrics.

            Yields:
                Streaming analysis outputs (text/Markdown) with referenced visuals.
            """
        detected_columns = self.extract_columns_from_query(query, self.eda_cleaned_df.columns.tolist())

        if len(detected_columns) < 1:
            yield "❌ Please specify at least one valid column."
            return

        if len(detected_columns) == 1:
            async for chunk in self.analyze_single_column(detected_columns[0], query):
                yield chunk
            return

        df = self.eda_cleaned_df.copy()
        time_cols = [col for col in df.columns if "date" in col.lower() or "start" in col.lower()]
        if time_cols:
            date_col = time_cols[0]
            try:
                df = df.sort_values(by=date_col)
                df = df.set_index(date_col)
            except Exception as e:
                print(f"⚠️ Can't convert date column: {e}")

        python_tool = PythonREPLTool()

        agent = create_pandas_dataframe_agent(
            self.chat_model,
            df,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            verbose=True,
            tools=[python_tool],
            callbacks=[StreamingStdOutCallbackHandler()]
        )

        try:
            response = await agent.ainvoke({"input": query})
            agent_analysis = response.get("output", "") if isinstance(response, dict) else str(response)
            if not agent_analysis.strip():
                yield "⚠️ Agent returned empty analysis."
                return
        except Exception as e:
            yield f"⚠️ Error occurred while running agent: {str(e)}"
            return

        try:
            lang = detect(query)
        except:
            lang = "en"

        retriever = Herakleitos.init_pg_rag_retriever(agent_name="analyze_agent", lang=lang)
        try:
            docs = retriever.get_relevant_documents(query)
            rag_context = "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            rag_context = ""
            print(f"⚠️ Failed to fetch RAG context: {e}")

        prompt_en = """
        🎯 **Your Role:** You are a data scientist with access to two inputs:

        📊 **Agent Analysis:**
        {agent_analysis}

        📚 **Contextual Knowledge from Domain:**
        {rag_context}

        ❓ Based on the above, explain the situation in light of the user's question: '{query}'.
        """.strip()

        prompt_tr = """
        🧠 **Rolün:** Bir veri bilimci olarak iki girdiye sahipsiniz:

        📊 **Veriden Elde Edilen Agent Analizi:**
        {agent_analysis}

        📚 **Alan Bilgisinden Gelen Bağlamsal Bilgi:**
        {rag_context}

        ❓ Yukarıdaki bilgilere göre '{query}' sorusunu yanıtlayınız.
        """.strip()

        chat_prompt, inputs = self.generate_prompt(
            query=query,
            prompt_tr=prompt_tr,
            prompt_en=prompt_en,
            lang=lang,
            agent_analysis=agent_analysis,
            rag_context=rag_context
        )

        chain = chat_prompt | self.chat_model

        async for chunk in chain.astream(inputs):
            yield chunk.content if hasattr(chunk, "content") else str(chunk)

    @traceable
    async def wiki_agent(self, query):
        """Answer knowledge questions via search tools (Wikipedia/DuckDuckGo/WolframAlpha).

            Args:
                query: Concept/definition question (e.g., “What is CTR?”).

            Yields:
                Short, sourced explanations suitable for quick reference.
            """
        if "WOLFRAM_ALPHA_APPID" not in os.environ:
            os.environ["WOLFRAM_ALPHA_APPID"] = config("WOLFRAM_ALPHA_APPID")


        wolfram_wrapper = WolframAlphaAPIWrapper()
        wolfram = WolframAlphaQueryRun(api_wrapper=wolfram_wrapper)
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        search = DuckDuckGoSearchRun()
        tools = [search, wikipedia, wolfram]

        tool_node = ToolNode(tools=tools)
        graph = StateGraph(MessagesState)



        llm_node = RunnableLambda(
            lambda state: {"messages": state["messages"] + [self.chat_model.invoke(state["messages"])]}
            )


        graph.add_node("llm", llm_node)
        graph.add_node("tools", tool_node)

        graph.set_entry_point("llm")
        graph.add_conditional_edges("llm", tools_condition)
        graph.add_edge("tools", "llm")
        graph.add_edge("llm", END)

        builder = graph.compile(checkpointer=self.checkpointer)

        config = {
            "configurable": {
                "thread_id": self.user_id,
            }
        }


        input_msg = HumanMessage(content=query)

        try:
            async for chunk in builder.astream({"messages": [input_msg]}, config, stream_mode="values"):
                latest = chunk["messages"][-1]
                content = getattr(latest, "content", None)
                if content:
                    cleaned = content.strip()
                    if cleaned.lower().startswith(query.lower()):
                        cleaned = cleaned[len(query):].lstrip(":").strip()
                    yield cleaned
        except Exception as e:
            print("❌ Wiki Agent Stream Exception:", str(e))
            yield "❌ Wikipedia agent stream failed. Please try again later."

    async def geo_agent(self, json_path, query):
        """Explain GeoLift power/impact results produced by ``powerAnalysis``.

            Args:
                json_path: Path to the model summaries JSON.
                query: User question that frames the interpretation.

            Returns:
                A natural-language summary of lift, power, and experimental design notes.
            """
        with open(json_path, "r") as file:
            geo_results = json.load(file)

        test_summary = geo_results["Test_Model"]
        best_summary = geo_results["Best_Model"]

        weight_table = "\n".join(
            f"| {w['location'].title()} | {w['weight']:.4f} | {bw['weight']:.4f} |"
            for w, bw in zip(test_summary['weights'], best_summary['weights'])
        )
        effect_timeline = json.dumps(best_summary["ATT"], indent=4)
        try:
            lang = detect(query)
        except:
            lang = "en"

        prompt_en = """
        🎯 **Your Role:** You are a data scientist. Please interpret the following results with the technical and analytical mindset of a data scientist.

        ## 📊 **GeoLift Experiment Analysis**

        🚀 **IMPORTANT: This is a GeoLift analysis designed to measure the causal impact of an advertising campaign.**
        We are testing whether **spending a budget of `{budget}`** led to a **significant lift** in `{y_id}`.

        **❗ READ CAREFULLY:**
        - **This is NOT a population analysis. DO NOT analyze location populations or city sizes.**
        - **The goal is to measure the causal effect of an advertising campaign using GeoLift.**
        - **We compare a Test Model (initial experiment) with a Best Model (optimized version with lowest imbalance).**
        - **We are looking for statistical evidence that the advertising spend increased `{y_id}`.**

        ---

        ## 📌 Experiment Summary
        - 📅 Treatment Period: `{start} to {end}`
        - 💰 Ad Budget Used: `{budget}`
        - 🔬 Experiment Type: `{type}`
        - 📈 Incremental Lift Observed: `{incremental}`
        - ⚖️ Bias Adjustment (Best Model): `{bias}`
        - 📊 L2 Imbalance Comparison (Test vs Best Model): `{l2_test} / {l2_best}`
        - 📊 Scaled L2 Imbalance: `{l2_scaled_test} / {l2_scaled_best}`
        - 🔍 Significance Level (Alpha): `{alpha}`

        ---

        ## 📎 Key Metrics Comparison

        | Metric                                      | Test Model              | Best Model             |
        |--------------------------------------------|--------------------------|------------------------|
        | **ATT Estimate (Effect Size)**             | `{att_est_test}`         | `{att_est_best}`       |
        | **Percentage Lift (Change in {y_id})**     | `{perc_lift_test}%`      | `{perc_lift_best}%`    |
        | **P-Value (Statistical Significance)**     | `{pvalue_test}`          | `{pvalue_best}`        |
        | **Incremental Effect**                     | `{incremental_test}`     | `{incremental_best}`   |

        📌 Interpretation Guidelines:
        - If p-value < 0.05, the effect is statistically significant.
        - If L2 imbalance is high, the models are not well-matched.
        - A high percentage lift indicates a successful campaign.

        ---

        ## 📍 Weight Distribution Across Locations

        | Location | Test Model Weight | Best Model Weight |
        |----------|-------------------|-------------------|
        {weight_table}

        ---

        ## 📊 Effect of Advertising Over Time

        ```json
        {effect_timeline}
        ```

        **🔎 Key Insights:**
        - **Look for time periods where the effect was strongest.**
        - **Check if the confidence intervals are narrow (meaning stable estimates) or wide (meaning high uncertainty).**
        - **If most time points have a p-value > 0.05, the effect is likely due to random variation rather than the ad campaign.**

        ---

        🎯 **Final Task: Summarize whether the advertising budget resulted in a meaningful lift in `{y_id}` and whether the experiment was statistically valid.**
        """

        prompt_tr = """
        🧠 **Rolün:** Sen bir veri bilimcisin. Lütfen aşağıdaki çıktıları, bir veri bilimcinin sahip olduğu teknik ve analitik bakış açısıyla değerlendir.

        ## 📊 GeoLift Deneyi Analizi

        🚀 **DİKKAT:** Bu analiz, bir reklam kampanyasının nedensel etkisini ölçmek için yapılmıştır.
        Amacımız, `{budget}` bütçesi harcandığında `{y_id}` değişkeninde anlamlı bir artış (lift) olup olmadığını test etmektir.

        ❗ **DİKKAT EDİLMESİ GEREKENLER:**
        - Bu bir nüfus analizi değildir. Lokasyon büyüklüklerine odaklanmayınız.
        - Amaç, reklam harcamasının nedensel etkisini ölçmektir.
        - Test modeli ile dengesizliği en düşük model karşılaştırılır.
        - `{y_id}` üzerindeki artış için istatistiksel kanıt aranır.

        ---

        ## 📌 Deney Özeti
        - 📅 Tedavi Dönemi: `{start} - {end}`
        - 💰 Reklam Bütçesi: `{budget}`
        - 🔬 Deney Türü: `{type}`
        - 📈 Gözlemlenen Artış: `{incremental}`
        - ⚖️ Bias Düzeltmesi: `{bias}`
        - 📊 L2 Dengesizlik (Test / Best): `{l2_test} / {l2_best}`
        - 📊 Ölçekli L2 Dengesizlik: `{l2_scaled_test} / {l2_scaled_best}`
        - 🔍 Anlamlılık Seviyesi (Alpha): `{alpha}`

        ---

        ## 📎 Temel Metrik Karşılaştırmaları

        | Metrik                                    | Test Modeli            | En İyi Model           |
        |------------------------------------------|------------------------|------------------------|
        | **ATT Tahmini (Etki Büyüklüğü)**         | `{att_est_test}`       | `{att_est_best}`       |
        | **Yüzdelik Artış ({y_id})**              | `{perc_lift_test}%`    | `{perc_lift_best}%`    |
        | **P-Değeri (Anlamlılık)**                | `{pvalue_test}`        | `{pvalue_best}`        |
        | **Artan Etki (Ek Kazanç/Etkileşim)**     | `{incremental_test}`   | `{incremental_best}`   |

        📌 Yorumlama Rehberi:
        - p-değeri < 0.05 ise, sonuç anlamlıdır.
        - L2 dengesizlik ne kadar düşükse, eşleştirme o kadar iyidir.
        - Yüksek yüzdelik artış kampanyanın başarılı olduğunu gösterir.

        ---

        ## 📍 Lokasyon Ağırlıkları

        | Lokasyon | Test Modeli Ağırlığı | En İyi Model Ağırlığı |
        |----------|----------------------|------------------------|
        {weight_table}

        ---

        ## 📊 Zaman İçinde Reklamın Etkisi

        ```json
        {effect_timeline}
        ```
        **🔎 Önemli İçgörüler:**
        - Etkinin en güçlü olduğu zaman aralıklarını inceleyin.  
        - Güven aralıkları dar ise bu, tahminlerin kararlı olduğunu gösterir; genişse belirsizlik yüksektir.  
        - Eğer çoğu zaman noktasında p-değeri > 0.05 ise, gözlenen etki reklam kampanyasından değil, rastlantısal dalgalanmalardan kaynaklanıyor olabilir.

        ---

        🎯 Son Görev: Reklam bütçesi, `{y_id}` değişkeninde anlamlı bir artış sağlamış mı? Ve bu deney istatistiksel olarak geçerli mi? Açıklayınız.
        """

        chat_prompt, inputs = self.generate_prompt(
            query=query,
            prompt_tr=prompt_tr,
            prompt_en=prompt_en,
            lang=lang,
            budget=best_summary.get("budget", "Unknown"),
            y_id=best_summary["Y_id"],
            start=best_summary["start"],
            end=best_summary["end"],
            type=best_summary["type"],
            incremental=best_summary["incremental"],
            bias=best_summary.get("bias", "N/A"),
            alpha=best_summary["alpha"],
            att_est_best=best_summary["ATT_est"],
            att_est_test=test_summary["ATT_est"],
            perc_lift_best=best_summary["PercLift"],
            perc_lift_test=test_summary["PercLift"],
            pvalue_best=best_summary["pvalue"],
            pvalue_test=test_summary["pvalue"],
            incremental_best=best_summary["incremental"],
            incremental_test=test_summary["incremental"],
            l2_test=test_summary["L2Imbalance"],
            l2_best=best_summary["L2Imbalance"],
            l2_scaled_test=test_summary["L2ImbalanceScaled"],
            l2_scaled_best=best_summary["L2ImbalanceScaled"],
            weight_table=weight_table,
            effect_timeline=effect_timeline
        )

        chain = chat_prompt | self.chat_model

        try:
            async for chunk in chain.astream(inputs):
                yield chunk.content if hasattr(chunk, "content") else str(chunk)
        except Exception as e:
            print("❌ LLM stream exception:", str(e))
            yield "❌ LLM model stream failed. Please try again later."


    async def eda_agent(self, json_files, df, col1, col2, query):
        """Summarize statistical tests and visuals between two columns.

            Args:
                json_files: Mapping of pair-type → JSON path (e.g., ``{"num_num": "...", ...}``).
                df: DataFrame used for EDA.
                col1: First column name.
                col2: Second column name.
                query: Analysis question guiding the narrative.

            Yields:
                Interpreted EDA findings (distributions, correlations, tests) in chunks.
            """
        is_col1_numeric = pd.api.types.is_numeric_dtype(df[col1])
        is_col2_numeric = pd.api.types.is_numeric_dtype(df[col2])

        is_col1_categorical = pd.api.types.is_categorical_dtype(df[col1]) or df[col1].dtype == "object"
        is_col2_categorical = pd.api.types.is_categorical_dtype(df[col2]) or df[col2].dtype == "object"

        if is_col1_numeric and is_col2_numeric:
            json_path = json_files["num_num"]
            analysis_type = "Numerical-Numerical Analysis"
        elif is_col1_numeric and is_col2_categorical or is_col1_categorical and is_col2_numeric:
            json_path = json_files["num_cat"]
            analysis_type = "Numerical-Categorical Analysis"
        elif is_col1_categorical and is_col2_categorical:
            json_path = json_files["cat_cat"]
            analysis_type = "Categorical-Categorical Analysis"
        else:
            yield f"⚠️ Could not determine the analysis type for {col1} and {col2}."
            return

        with open(json_path, "r") as file:
            analysis_results = json.load(file)

        col1_levels = df[col1].unique().tolist() if is_col1_categorical else "N/A"
        col2_levels = df[col2].unique().tolist() if is_col2_categorical else "N/A"

        graph_results = {}
        graph_path = "plots/scientific_image_analysis.json"
        if os.path.exists(graph_path):
            with open(graph_path, "r") as file:
                graph_results = json.load(file)

        graph_section_en = f"""
        #### **Graphical Analysis Insights**
        ```json
        {json.dumps(graph_results, indent=4)}
        ```""" if graph_results else ""

        prompt_en = """
        📘 Note:
        The ISKIN Score is a custom statistical measure designed to detect complex and non-linear relationships between numerical variables. It ranges from 0 to 1, where higher scores indicate stronger relationships. Unlike traditional correlation tests (e.g., Kendall or Pearson), ISKIN may detect subtle dependencies that do not manifest linearly. A score above 0.9 is typically considered strong.

        🎯 **Your Role:** You are a data scientist. Please interpret the following results with the technical and analytical mindset of a data scientist.
        ### 📊 Exploratory Data Analysis (EDA)

        **Analyzing the relationship between:** `{col1}` and `{col2}`
        - **Analysis Type:** {analysis_type}
        - **Data Types:** `{col1}` ({col1_type}), `{col2}` ({col2_type})
        - **Levels for Categorical Columns:**
        - `{col1}`: {col1_levels}
        - `{col2}`: {col2_levels}

        #### **Raw JSON Data:**
        ```json
        {analysis_results}
        ```

        ❓ **Interpretation Request**
        1️⃣ **Analyze the statistical test results.**  
        2️⃣ **Based on the p-values, determine if there is a significant relationship.**  
        3️⃣ **Explain the effect sizes and power of the tests.**  
        4️⃣ **Compare different statistical tests and identify inconsistencies.**  
        5️⃣ **Provide actionable insights based on the results.**  
        {graph_section_en}
        """.strip()

        graph_section_tr = f"""
        #### **Grafiksel Analiz Bulguları**
        ```json
        {json.dumps(graph_results, indent=4, ensure_ascii=False)}
        ```""" if graph_results else ""

        prompt_tr = """
        🧠 **Rolün:** Sen bir veri bilimcisin. Lütfen aşağıdaki çıktıları, bir veri bilimcinin sahip olduğu teknik ve analitik bakış açısıyla değerlendir.    
        ### 📊 Keşifsel Veri Analizi (EDA)

        **İncelenen ilişki:** `{col1}` ile `{col2}` arasındaki ilişki  
        - **Analiz Türü:** {analysis_type}  
        - **Veri Türleri:** `{col1}` ({'Sayısal' if is_col1_numeric else 'Kategorik'}), `{col2}` ({'Sayısal' if is_col2_numeric else 'Kategorik'})  
        - **Kategorik Değişkenlerin Seviyeleri:**
        - `{col1}`: {col1_levels}
        - `{col2}`: {col2_levels}

        #### **Ham JSON Verisi:**
        ```json
        {analysis_results}
        ```
        ❓ **Yorumlama Talebi**  
        1️⃣ **İstatistiksel test sonuçlarını analiz edin.**  
        2️⃣ **p-değerlerine göre anlamlı bir ilişki olup olmadığını belirleyin.**  
        3️⃣ **Etki büyüklüklerini (effect size) ve test gücünü (statistical power) açıklayın.**  
        4️⃣ **Farklı istatistiksel testleri karşılaştırın ve varsa tutarsızlıkları belirleyin.**  
        5️⃣ **Sonuçlara dayanarak uygulanabilir içgörüler sunun.**
        {graph_section_tr}
        """.strip()
        try:
            lang = detect(query)
        except:
            lang = "en"

        chat_prompt, prompt_inputs = self.generate_prompt(
            query=query,
            prompt_tr=prompt_tr,
            prompt_en=prompt_en,
            lang=lang,
            col1=col1,
            col2=col2,
            col1_type="Numeric" if is_col1_numeric else "Categorical",
            col2_type="Numeric" if is_col2_numeric else "Categorical",
            col1_type_tr="Sayısal" if is_col1_numeric else "Kategorik",
            col2_type_tr="Sayısal" if is_col2_numeric else "Kategorik",
            analysis_type=analysis_type,
            col1_levels=col1_levels,
            col2_levels=col2_levels,
            analysis_results=analysis_results,
            graph_section_en=graph_section_en,
            graph_section_tr=graph_section_tr
        )

        chain = chat_prompt | self.chat_model

        try:
            async for chunk in chain.astream(prompt_inputs):
                yield chunk.content if hasattr(chunk, "content") else str(chunk)
        except Exception as e:
            print("❌ LLM stream exception:", str(e))
            yield "❌ LLM model stream failed. Please try again later."

    async def ab_agent(self, user_query):
        """Analyze A/B tests and report significance across methods.

            Args:
                user_query: Experiment question (variants, metric, horizon).

            Yields:
                Test summaries with p-values, effect sizes, and decision guidance.
            """
        with open('services/AI/data/ab_test.json', 'r') as file:
            ab_test_results = json.load(file)

        prompt = f"""
            🎓 Note:
            You are a highly experienced data scientist and statistician with over 30 years of expertise in experimental design and A/B testing methodologies. You are asked to provide a detailed interpretation of an A/B testing analysis.

            ### 📝 User Query:
            "{user_query}"

            ### 📊 A/B Testing Results:
            The following JSON represents the results of various A/B testing methodologies (e.g., Z-Test, Fisher's Exact Test, T-Test, Chi-Square Test, ANOVA). Each result includes key metrics like p-values, effect sizes, and significance levels.

            ```json
            {ab_test_results}
            ```

            #### 🎯 Your Analytical Task:
            1️⃣ **Analyze the test results** — Describe what the p-values, test types, and effect sizes indicate about the performance difference between groups.
            2️⃣ **Assess Statistical Significance** — For each test, determine if the results are statistically significant based on the p-value thresholds.
            3️⃣ **Interpret Effect Sizes** — Explain how large or meaningful the observed effect sizes are in practical/business terms.
            4️⃣ **Compare Methods** — Identify if there are discrepancies between different statistical methods (e.g., Fisher vs Z-Test), and explain why such inconsistencies may occur.
            5️⃣ **Provide Data-Driven Recommendations** — Based on these results, suggest actionable next steps (e.g., scale experiment, gather more data, validate findings).
            6️⃣ **Mention Limitations** — Highlight any limitations or assumptions inherent to the test results (e.g., sample size concerns, test type appropriateness).

            🎓 You are expected to give a structured, precise, and scientific response.
            """.strip()

        try:
            async for chunk in self.chat_model.astream(prompt):
                yield chunk.content if hasattr(chunk, "content") else str(chunk)
        except Exception as e:
            print("❌ LLM stream exception:", str(e))
            yield "❌ LLM model stream failed. Please try again later."

    async def assoc_agent(self, user_query: str, json_path: str = "data/assoc_rules.json", top_n: int = 50):
        """Stream an LLM-based interpretation of association rules.

            Loads a JSON file containing association rules (e.g., Apriori, FP-Growth),
            trims to the top-N rules by lift/confidence/support, and builds a structured
            interpretation prompt. The prompt guides the LLM to summarize patterns,
            highlight entity-linked vs. metric-only rules, and produce actionable insights.

            Args:
                user_query: Natural language question or focus area for interpretation.
                json_path: Path to the JSON file containing precomputed association rules.
                top_n: Number of top-ranked rules to include (default = 50).

            Yields:
                Streaming text chunks with:
                  - Summarized association patterns (ranked by lift, confidence, support)
                  - Entity-specific vs. general metric findings
                  - Conflicts, trivialities, and caveats
                  - Actionable recommendations for campaigns or experiments
                  - Limitations and notes on interpretability
            """
        # 1) Load JSON
        if not os.path.exists(json_path):
            yield f"❌ JSON not found at {json_path}"
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                rules_obj = json.load(f)
        except Exception as e:
            yield f"❌ JSON load error: {e}"
            return

        # 2) Optional top_n trimming for large files
        trimmed = rules_obj
        if isinstance(rules_obj, list) and top_n and top_n > 0:
            def _as_num(x):
                try:
                    return float(x)
                except:
                    return -1e9

            def sort_key(r):
                return (-_as_num(r.get("lift", 0)),
                        -_as_num(r.get("confidence", 0)),
                        -_as_num(r.get("support", 0)))

            trimmed = sorted(rules_obj, key=sort_key)[:top_n]

        try:
            rules_json_str = json.dumps(trimmed, ensure_ascii=False, indent=2)
        except Exception:
            rules_json_str = str(trimmed)

        # 3) Prompt for Association Rules interpretation
        prompt = f"""
    🎓 Note:
    You are a senior data scientist specializing in **Association Rule Mining** (Apriori / FP-Growth / ECLAT) for marketing analytics. 
    Your task is to provide a clear, precise, and business-focused interpretation of the given rules.

    ### 📝 User Query
    "{user_query}"

    ### 📊 Association Rules (JSON)
    Below is the rules output (top {top_n} shown if large):
    ```json
    {rules_json_str}
    📘 Quick Primer (interpretation guide)
    Antecedents → Consequents: Read as "If antecedents, then consequents".

    support = P(A ∧ B): Frequency of both conditions occurring together. Higher support = more common pattern.

    confidence = P(B|A): Probability of B given A. Ranges 0–1.

    lift = P(B|A) / P(B): >1 = positive association, ≈1 = no association, <1 = negative association.

    antecedent support / consequent support: P(A) and P(B) individually.

    leverage = P(A∧B) − P(A)P(B): Distance from independence. Larger magnitude = stronger deviation.

    conviction = (1−P(B)) / (1−P(B|A)): Higher means B is less likely to be absent when A occurs.

    Entity items (e.g., campaign_name:…, adset_name:…, ad_name:…, Level:…) show rules tied to specific campaigns/ad sets/ads.

    Binning (low/mid/high): Metrics (CTR, CPC, SPEND, IMPR, CONV) are bucketed; interpret relatively.

    🎯 Your Analytical Task
    Core findings: Summarize key patterns, prioritizing by lift, then confidence, then support.

    Entity-aware insights: Identify which campaigns/adsets/ads are linked to positive patterns (e.g., CTR:high, CPC:low) or negative ones (CTR:low, CPC:high).

    Metric-only vs entity-linked: Distinguish between general metric rules and entity-specific ones.

    Trivialities & conflicts: Downrank obvious relationships (e.g., SPEND:high → IMPR:high). Note conflicting outcomes from same antecedents and explain potential causes.

    Actionable recommendations: Suggest campaign budget shifts, creative/targeting changes, A/B tests, or follow-up analysis. Prioritize actions.

    Limitations: Note binning artifacts, low support, imbalanced classes, limited entity coverage, and that association ≠ causation. Suggest possible data improvements.

    Please answer in clear sections with bullet points, keeping the response concise but insight-rich.
    """.strip()

        try:
            async for chunk in self.chat_model.astream(prompt):
                yield chunk.content if hasattr(chunk, "content") else str(chunk)
        except Exception as e:
            print("❌ LLM stream exception:", str(e))
            yield "❌ LLM model stream failed. Please try again later."



    async def shap_agent(self, col, query):
        """Produce SHAP interpretability narrative for a trained regression model.

            Args:
                col: Target variable analyzed with SHAP.
                query: User question to focus the explanation.

            Yields:
                Explanations of global/local importance and interaction hints.
            """
        with open("services/AI/data/best_regression_results.json", "r") as file:
            result_model = json.load(file)

        with open("services/AI/data/final_result_regress.json", "r") as file:
            result_shap = json.load(file)

        prompt_en = """
        🎯 **Your Role:** You are a data scientist. Please interpret the following results with the technical and analytical mindset of a data scientist.
        ### 📊 Regression Agent Analysis

        **{query}:**  
        - **Target Variable:** `{col}`  
        - **Best Regression Model:** `{best_model}`  
        - **Evaluation Metrics:**  
            - `R²`: {best_r2}  
            - `MAE`: {best_mae}  
            - `RMSE`: {best_rmse}  

        #### **Suggested Feature Contributions (SHAP Score Analysis)**  
        ```json
        {shap_scores}
        ```

        ❓ **Interpretation Request**  
        1️⃣ Analyze how each feature contributes to the target change.  
        2️⃣ Determine the significance using SHAP values.  
        3️⃣ Compare feature rankings and highlight key drivers.  
        4️⃣ Assess the reliability of predictions.  
        5️⃣ Suggest actionable strategies.  
        """.strip()

        prompt_tr = """
        🧠 **Rolün:** Sen bir veri bilimcisin. Aşağıdaki çıktıları yorumla:  
        ### 📊 Regresyon Ajanı Analizi

        **{query}:**  
        - **Hedef Değişken:** `{col}`  
        - **En İyi Regresyon Modeli:** `{best_model}`  
        - **Değerlendirme Metrikleri:**  
            - `R²`: {best_r2}  
            - `MAE`: {best_mae}  
            - `RMSE`: {best_rmse}  

        #### **Özellik Katkıları (SHAP Skorları)**  
        ```json
        {shap_scores}
        ```

        ❓ **Yorumlama Talebi:**  
        1️⃣ Hangi değişken ne kadar katkı sağladı?  
        2️⃣ SHAP değeriyle önem derecesini belirt.  
        3️⃣ Etkili değişkenleri sırala.  
        4️⃣ Tahmin güvenilirliğini değerlendir.  
        5️⃣ Stratejik önerilerde bulun.
        """.strip()

        try:
            lang = detect(query)
        except:
            lang = "en"

        chat_prompt, prompt_inputs = self.generate_prompt(
            query=query,
            prompt_tr=prompt_tr,
            prompt_en=prompt_en,
            lang=lang,
            col=col,
            best_model=result_model["best_model"],
            best_r2=result_model["best_r2"],
            best_mae=result_model["best_mae"],
            best_rmse=result_model["best_rmse"],
            shap_scores=json.dumps(result_shap, indent=4, ensure_ascii=(lang != "tr"))
        )

        chain = chat_prompt | self.chat_model
        try:
            async for chunk in chain.astream(prompt_inputs):
                yield chunk.content if hasattr(chunk, "content") else str(chunk)
        except Exception as e:
            print("❌ LLM stream exception:", str(e))
            yield "❌ LLM model stream failed. Please try again later."




    async def ts_agent(self, user_query: str, dashboard=None):
        """Explain time-series forecasts/causality with optional dashboard rendering.

            Args:
                user_query: Forecasting/causality question (may contain horizon or series).
                dashboard: Optional dashboard object to render or export.

            Yields:
                Forecast summaries, uncertainty notes, and causal diagnostics.
            """
        try:
            with open("services/AI/data/ts_metrics.json", "r") as file:
                results = json.load(file)

            chat_prompt = PromptTemplate.from_template("""
    📌 **Your Role:** You are a data scientist assigned to interpret the results of a time series and causal analysis.

    You are provided with the following data (may include forecasting and/or causality information):

    🧾 **Analysis Result**:
    {analysis_result}

    🗣️ **User Query**:
    {user_query}

    ---

    ✅ Based on the data above, answer the following:

    1. If forecast results are included:
       - Are the predictions statistically reliable? (e.g. MAE, MAPE, R²)
       - Are there major trend shifts or changepoints?
       - Do holidays have a visible influence on predictions?

    2. If multiple metrics are analyzed:
       - Which metrics show strongest trends or weakest prediction accuracy?
       - Are there any that should be prioritized, flagged, or adjusted?

    3. If Granger causality information is present:
       - Is there statistical causality between metrics?
       - What does the p-value and optimal lag indicate?
       - Is this relationship useful for decision-making?

    4. Finally, considering the user query, write a narrative interpretation and provide concrete, business-relevant insights or recommendations.
    """)

            chain = chat_prompt | self.chat_model

            prompt_inputs = {
                "analysis_result": json.dumps(results, indent=2),
                "user_query": user_query
            }

            async for chunk in chain.astream(prompt_inputs):
                yield chunk.content if hasattr(chunk, "content") else str(chunk)

            if dashboard is not None:
                dashboard_name = "forecast_dashboard"
                dashboard.servable("Forecast Dashboard")
                dashboard_url = f"http://localhost:5006/{dashboard_name}"
                yield f"\n📊 [Open Forecast Dashboard]({dashboard_url})"
        except Exception as e:
            print("❌ LLM stream exception:", str(e))
            yield "❌ LLM model stream failed. Please try again later."

    async def dowhy_agent(self, json_file, query):
        """Interpret DoWhy/EconML causal outputs from persisted JSON.

            Args:
                json_file: Path to the causal results JSON.
                query: Causal question (treatment/outcome scope).

            Returns:
                A human-readable summary of effects, assumptions, and robustness checks.
            """
        with open(json_file, "r") as file:
            analysis_results = json.load(file)

        analysis_file = "services/AI/data/dowhy_econml_causal.json"
        graph_results = {}
        if os.path.exists(analysis_file):
            with open(analysis_file, "r") as file:
                graph_results = json.load(file)

        prompt_en = """
        🎯 **Your Role:** You are a data scientist. Please interpret the following results with the technical and analytical mindset of a data scientist.
        ### 🔍 **Causal Inference Analysis Using DoWhy & EconML**

        📝 **User Query:** "{query}"

        - The system has conducted a **causal analysis** using DoWhy and EconML.
        - The dataset has been processed, and the estimated **causal effects** are extracted.
        - The results include **treatment effects, statistical refutations, and graphical model insights**.

        #### **📊 Causal Inference Results**
        ```json
        {analysis_results}
        ```

        #### **📉 Graphical Analysis Insights**
        ```json
        {graph_results}
        ```

        ❓ **Interpretation Request**
        1️⃣ Explain the estimated causal effects in plain terms.  
        2️⃣ Analyze how the treatment variable influences the outcome variable.  
        3️⃣ Evaluate whether the model assumptions are valid.  
        4️⃣ Identify potential biases or inconsistencies in the analysis.  
        5️⃣ Compare different causal estimation methods used in the analysis.  
        6️⃣ Suggest improvements or refinements for a more robust causal inference.
        """.strip()

        prompt_tr = """
        🧠 **Rolün:** Sen bir veri bilimcisin. Lütfen aşağıdaki çıktıları, bir veri bilimcinin sahip olduğu teknik ve analitik bakış açısıyla değerlendir.   
        ### 🔍 **DoWhy & EconML Kullanarak Nedensel Çıkarım Analizi**

        📝 **Kullanıcı Sorgusu:** "{query}"

        - Sistem, DoWhy ve EconML kütüphanelerini kullanarak bir **nedensel analiz** gerçekleştirmiştir.  
        - Veri kümesi işlenmiş ve tahmin edilen **nedensel etkiler** çıkarılmıştır.  
        - Sonuçlar arasında **tedavi (treatment) etkileri, istatistiksel doğrulamalar ve grafiksel model içgörüleri** yer almaktadır.

        #### **📊 Nedensel Çıkarım Sonuçları**
        ```json
        {analysis_results}
        ```

        #### **📉 Grafiksel Analiz İçgörüleri**
        ```json
        {graph_results}
        ```

        ❓ **Yorumlama Talebi**  
        1️⃣ Tahmin edilen nedensel etkileri sade ve anlaşılır bir dille açıklayınız.  
        2️⃣ Tedavi (müdahale) değişkeninin sonuç (bağımlı) değişken üzerindeki etkisini analiz ediniz.  
        3️⃣ Model varsayımlarının geçerliliğini değerlendiriniz.  
        4️⃣ Analizdeki olası önyargı (bias) veya tutarsızlıkları belirleyiniz.  
        5️⃣ Kullanılan farklı nedensel tahmin yöntemlerini karşılaştırınız.  
        6️⃣ Daha sağlam bir nedensel çıkarım için iyileştirme veya geliştirme önerileri sununuz.
        """.strip()

        try:
            lang = detect(query)
        except:
            lang = "en"

        chat_prompt, prompt_inputs = self.generate_prompt(
            query=query,
            prompt_tr=prompt_tr,
            prompt_en=prompt_en,
            lang=lang,
            analysis_results=json.dumps(analysis_results, indent=4, ensure_ascii=(lang != "tr")),
            graph_results=json.dumps(graph_results, indent=4, ensure_ascii=(lang != "tr"))
        )

        chain = chat_prompt | self.chat_model
        try:
            async for chunk in chain.astream(prompt_inputs):
                yield chunk.content if hasattr(chunk, "content") else str(chunk)
        except Exception as e:
            print("❌ LLM stream exception:", str(e))
            yield "❌ LLM model stream failed. Please try again later."

    async def causalpy_agent(self, json_file, query):
        """Interpret CausalPy synthetic-control (Sklearn/Bayesian) outputs.

            Args:
                json_file: Path to synthetic control summaries JSON.
                query: User question to contextualize the causal impact.

            Returns:
                Narrative explaining pre/post fit, counterfactuals, and estimated impact.
            """
        with open("services/AI/data/sklearn_and_bayesian_causal_impact_summary.json", "r") as file:
            analysis_results = json.load(file)

        graph_results = {}
        if os.path.exists(json_file):
            with open(json_file, "r") as file:
                graph_results = json.load(file)

        try:
            lang = detect(query)
        except:
            lang = "en"

        prompt_en = """
        🎯 **Your Role:** You are a data scientist. Please interpret the following results with the technical and analytical mindset of a data scientist.
        ### 🔍 **Causal Inference Analysis Using CausalPy**

        📝 **User Query:** "{query}"

        - The system has conducted a **causal analysis** using CausalPy.
        - The dataset has been processed using both **Bayesian (PyMC)** and **Sklearn-based** Synthetic Control models.
        - The results include **causal impact estimates, model fit scores, and graphical model insights**.

        #### **📊 Bayesian & Sklearn-Based Causal Analysis Results**
        ```json
        {analysis_results}
        ```

        #### **📉 Graphical Analysis Insights**
        ```json
        {graph_results}
        ```

        ❓ **Interpretation Request**

        Please provide a structured and in-depth interpretation of the results based on the following points:
        
        ---
        
        ### 1️⃣ **Comparison: Bayesian vs. Sklearn-Based Models**
        - How do the causal effect estimates differ between the two models?
        - Which model demonstrates a better pre-intervention fit to the observed data?
        - Does the Bayesian model offer better uncertainty quantification (e.g., credible intervals)?
        - Are the model assumptions consistent across both methods?
        
        ---
        
        ### 2️⃣ **Evaluation of Causal Impact Results**
        - Is the estimated treatment effect statistically and practically significant?
        - How closely does the counterfactual prediction match the actual observations?
        - Are there any deviations or surprises in the post-intervention period?
        - Is there evidence of delayed or cumulative treatment effects?
        
        ---
        
        ### 3️⃣ **Model Robustness and Bias Assessment**
        - Does either model show signs of overfitting or instability?
        - How sensitive are the results to changes in predictor variables or treatment timing?
        - Are the causal estimates stable across different time frames or subgroups?
        
        ---
        
        ### 4️⃣ **Graphical Analysis Insights**
        - How do the pre-intervention fit, causal impact plot, and cumulative impact chart support the conclusions?
        - Are the trends in the graphical outputs consistent with numerical estimates?
        - Do the graphs highlight any anomalies, structural breaks, or unexpected outliers?
        - How do visual elements improve understanding of model performance and reliability?
        
        ---
        
        ### 5️⃣ **Scientific Insights and Recommendations**
        - What is the overall conclusion regarding the presence and magnitude of causal impact?
        - Which model is more appropriate for making business or policy decisions?
        - What additional checks (e.g., placebo tests, sensitivity analysis) could strengthen the findings?
        - Are there alternative causal inference methods you would recommend testing?
        
        ---
        
        📌 Present your answer as if you are preparing a data science report for a technical audience. Include justifications, references to visual cues, and actionable recommendations based on the analysis.
        
        """.strip()

        prompt_tr = """
        🧠 **Rolün:** Sen bir veri bilimcisin. Lütfen aşağıdaki çıktıları, bir veri bilimcinin sahip olduğu teknik ve analitik bakış açısıyla değerlendir.  
        ### 🔍 **CausalPy Kullanarak Nedensel Çıkarım Analizi**

        📝 **Kullanıcı Sorgusu:** "{query}"

        - Sistem, **CausalPy** kullanarak bir **nedensel analiz** gerçekleştirmiştir.  
        - Veri kümesi, hem **Bayesyen (PyMC)** hem de **Sklearn tabanlı** Sentetik Kontrol modelleriyle işlenmiştir.  
        - Sonuçlar, **nedensel etki tahminleri**, **model uyum skorları** ve **grafiksel model içgörüleri** içermektedir.

        #### **📊 Bayesyen ve Sklearn Tabanlı Nedensel Analiz Sonuçları**
        ```json
        {analysis_results}
        ```

        #### **📉 Grafiksel Analiz İçgörüleri**
        ```json
        {graph_results}
        ```

        1️⃣ Bayesyen ve Sklearn Tabanlı Modelleri Karşılaştırınız:
            - Tahmin edilen nedensel etkiler arasında nasıl farklar var?
            - Hangi model müdahale öncesi dönemde daha iyi bir uyum sağlıyor?
            - Bayesyen çıkarım, belirsizliklerin ölçümünde daha başarılı mı?

        2️⃣ Nedensel Etki Sonuçlarını Değerlendiriniz:
            - Tahmin edilen tedavi etkisi ne kadar anlamlı?
            - Karşı olgusal (counterfactual) tahminler beklentilerle örtüşüyor mu?
            - Müdahale sonrası dönemde ciddi sapmalar var mı?

        3️⃣ Model Sapmaları ve Sınırlılıkları Üzerine Analiz:
            - Modellerden biri aşırı öğrenmeye (overfitting) maruz kalıyor mu?
            - Farklı tahminci seçimlerine karşı sonuçlar ne kadar sağlam?
            - Nedensel etkiler zaman boyunca istikrarlı mı?

        4️⃣ Bilimsel İçgörüler ve Önerilen Geliştirmeler:
            - Hangi yöntem karar verme açısından daha güvenilirdir?
            - Ek sağlamlık kontrolleri yapılmalı mı?
            - Test edilebilecek alternatif nedensel tahmin yöntemleri nelerdir?

        5️⃣ Grafiksel İçgörüler ve Yorumlama:
            - Müdahale öncesi uyum, nedensel etki grafiği ve kümülatif nedensel etki grafiklerinin yoruma katkısı nedir?
            - Grafiksel çıktılardaki eğilimler, sayısal nedensel tahminlerle tutarlı mı?
            - Görseller, sayısal çıktılarda görülmeyen önyargılar ya da aykırı değerler ortaya koyuyor mu?
            - Grafiksel bulgular, nedensel çıkarım sonuçlarının güvenilirliğini ve sağlamlığını nasıl etkiler?

        📌 Lütfen yanıtınızı açık bilimsel açıklamalarla ve net çıkarımlarla yapılandırılmış şekilde sununuz.
        """.strip()

        chat_prompt, prompt_inputs = self.generate_prompt(
            query=query,
            prompt_tr=prompt_tr,
            prompt_en=prompt_en,
            lang=lang,
            analysis_results=json.dumps(analysis_results, indent=4, ensure_ascii=(lang != "tr")),
            graph_results=json.dumps(graph_results, indent=4, ensure_ascii=(lang != "tr"))
        )

        chain = chat_prompt | self.chat_model

        try:
            async for chunk in chain.astream(prompt_inputs):
                yield chunk.content if hasattr(chunk, "content") else str(chunk)
        except Exception as e:
            print("❌ LLM stream exception:", str(e))
            yield "❌ LLM model stream failed. Please try again later."



    @staticmethod
    def filter_data_last_n_days(data, n_days):
        """Filter dataset to the last ``n_days`` based on available date columns.

            Args:
                data: Input DataFrame.
                n_days: Number of trailing days to keep.

            Returns:
                Filtered DataFrame restricted to the latest period.
            """
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            latest_date = df['date'].max()
            earliest_date = df['date'].min()

            max_available_days = (latest_date - earliest_date).days
            if n_days > max_available_days:
                return None, f"⚠️ Warning: The dataset only contains data for the last {max_available_days} days. Please adjust your query."

            cutoff_date = latest_date - timedelta(days=n_days)
            filtered_df = df[df['date'] >= cutoff_date]

            if filtered_df.empty:
                return None, f"⚠️ Warning: No data available for the last {n_days} days."

            return filtered_df, None

        start_cols = [col for col in df.columns if re.search(r'\b(starts|begins)\b', col.lower())]
        end_cols = [col for col in df.columns if re.search(r'\b(ends|finishes)\b', col.lower())]

        if not start_cols or not end_cols:
            return None, "⚠️ Warning: No valid date column ('date', 'start-*', 'end-*') found in the dataset."

        start_col = start_cols[0]
        end_col = end_cols[0]

        df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
        df[end_col] = pd.to_datetime(df[end_col], errors='coerce')

        latest_end = df[end_col].max()
        earliest_start = df[start_col].min()

        max_available_days = (latest_end - earliest_start).days
        if n_days > max_available_days:
            return None, f"⚠️ Warning: The maximum available date range is {max_available_days} days. Please adjust your query."

        cutoff_date = latest_end - timedelta(days=n_days)

        filtered_df = df[(df[start_col] >= cutoff_date) & (df[end_col] <= latest_end)]

        if filtered_df.empty:
            return None, f"⚠️ Warning: No data available for the last {n_days} days."

        return filtered_df, None

    def extract_columns_from_query(self, query, available_columns):
        """Detect potential column names mentioned in a natural-language query.

            Args:
                query: Free-form user text.
                available_columns: List of column names to match against.

            Returns:
                A list of best-guess column names (may be empty if none found).
            """
        words = re.findall(r'\w+', query.lower())
        seen = set()
        detected_columns = []

        for word in words:
            if len(word) < 2:
                continue

            match = self.find_best_column_match(word, available_columns)
            if match and match not in seen:
                detected_columns.append(match)
                seen.add(match)

        detected_columns = [col for col in detected_columns if col in available_columns]
        return detected_columns

    def extract_column_name(self, query, field_name):
        """Extract a column referred to as ``as <FIELD_NAME>`` in the query.

            Args:
                query: User text potentially containing the pattern.
                field_name: Logical field label to search for (e.g., ``"metric"``).

            Returns:
                Extracted column name or ``None`` if not present.
            """
        import re

        # Sadece 'X as FIELD' eşleşmeleri yakalanır
        pattern = rf"(\w+)\s+as\s+{field_name}"
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            candidate = match.group(1).lower()
            stopwords = {"and", "as", "with", "or", "not"}
            if candidate in stopwords:
                return None
            return candidate
        return None

    @staticmethod
    def extract_days_from_query(query):
        """Parse phrases like “last N days” from the user query.

            Args:
                query: Input text.

            Returns:
                Integer number of days if detected; otherwise ``None``.
            """
        match = re.search(r'(?:last\s*)?(\d+)\s*(?:days|day)', query, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def generate_plots(self, data, columns):
        """Generate appropriate plots based on column-type combinations.

            Args:
                data: Source DataFrame used to draw figures.
                columns: Iterable of column pairs to visualize.

            Returns:
                A collection of figure objects or file paths to saved plots.
            """
        data = pd.DataFrame(data)
        for col1, col2 in columns:
            if col1 in data.columns and col2 in data.columns:
                if os.path.exists("data/num_num.json"):
                    with open("data/num_num.json") as file:
                        result = json.loads(file.read())
                    correlation_method = next(iter(result.keys())).lower()
                else:
                    edaAnalysis(self.eda_cleaned_df, col1, col2)
                    with open("data/num_num.json") as file:
                        result = json.loads(file.read())
                    correlation_method = next(iter(result.keys())).lower()

                return all_graphs(data, col1, col2, correlation_method)

    async def determine_query_type(self, query):
        """Classify the user query into an analysis category.

            Uses LLM-based streaming classification with keyword fallbacks.

            Args:
                query: End-user prompt.

            Returns:
                One of ``{"eda","analyze","shap","geolift","dowecon","causalpy",
                "ts","wiki","plot","ab","assoc","unknown"}``.
            """
        classification_prompt = f"""
        You are a smart AI assistant that classifies user queries into one of the following categories:

        - eda → Summary statistics, distributions, correlation tests, variable types
        - analyze → Metric behavior over time, performance trends, metric-level changes
        - shap → SHAP values, feature importance, model interpretability
        - geolift → Regional or geo-based impact, marketing uplift
        - dowEcon → DoWhy/EconML causal inference, DAG-based reasoning
        - causalPy → Bayesian or time-series causal analysis (e.g. synthetic control)
        - ts → Time series analysis, Prophet, forecasting, seasonal decomposition
        - wiki → General knowledge or definitions (e.g. "What is CTR?")
        - plot → Requests for visualizations or chart generation
        - ab → A/B testing, experiment comparison, statistical significance tests
        - assoc → Association rules, market basket analysis, frequent itemset mining

        Your task: Based on the user query, return ONLY the category name (`eda`, `analyze`, etc.) with no explanation or formatting.

        User query:
        \"{query}\"

        Your answer:
        """

        collected = ""
        async for chunk in self.chat_model.astream(classification_prompt):
            collected += chunk.content

        response = collected.strip().lower()

        valid_categories = {
            "eda", "analyze", "shap", "geolift", "dowecon", "causalpy", "ts", "wiki", "plot", "ab", "assoc"
        }

        for cat in valid_categories:
            if cat in response:
                return cat

        # Fallback keyword-based detection for AB Testing
        ab_keywords = ["a/b test", "ab test", "split test", "ab testing", "experiment comparison", "significance test"]
        if any(kw in query.lower() for kw in ab_keywords):
            return "ab"

        # Fallback keyword-based detection for Association Rules
        assoc_keywords = [
            "association rules", "market basket", "apriori", "fpgrowth", "fp-growth",
            "frequent itemset", "lift", "confidence", "support"
        ]
        if any(kw in query.lower() for kw in assoc_keywords):
            return "assoc"

        if self.parse_causal_query(query):
            return "dowecon"
        elif self.parse_causalpy_query(query):
            return "causalpy"
        elif "time series" in query.lower() or "forecast" in query.lower():
            return "ts"

        return "unknown"


    @staticmethod
    def normalize(text):
        """Lowercase and strip special characters for robust matching.

            Args:
                text: Input string.

            Returns:
                Normalized string suitable for comparisons.
            """
        return re.sub(r'[^a-zA-Z0-9]', '', text.lower())

    def find_best_column_match(self, user_column, available_columns, verbose=True):
        """Fuzzy-match the requested column name to the closest available one.

            Args:
                user_column: Column name as provided by the user.
                available_columns: Valid columns in the dataset.

            Returns:
                The best-matching column name, or ``None`` if no acceptable match.
            """
        user_column_cleaned = self.normalize(user_column)

        column_mapping = {
            self.normalize(col): col for col in available_columns
        }

        if user_column_cleaned in column_mapping:
            matched = column_mapping[user_column_cleaned]
            if verbose:
                print(f"✅ [EXACT MATCH] '{user_column}' → '{matched}'")
            return matched

        for norm_col, original_col in column_mapping.items():
            if user_column_cleaned in norm_col or norm_col.startswith(user_column_cleaned):
                if verbose:
                    print(f"🔎 [SUBSTRING MATCH] '{user_column}' → '{original_col}'")
                return original_col

        closest_match = difflib.get_close_matches(
            user_column_cleaned, column_mapping.keys(), n=1, cutoff=0.85
        )
        if closest_match:
            matched = column_mapping[closest_match[0]]
            if verbose:
                print(f"🌀 [FUZZY MATCH] '{user_column}' → '{matched}' (score ≈ close)")
            return matched

        if verbose:
            print(f"❌ [NO MATCH] '{user_column}' → None")

        return None

    @staticmethod
    def validate_columns(cols, cleaned_df, raw_df=None):
        """Validate that requested columns exist in cleaned/raw datasets.

            Args:
                cols: Column names to validate.
                cleaned_df: Cleaned DataFrame.
                raw_df: Optional raw DataFrame for cross-checks.

            Returns:
                ``None`` if all columns are valid; otherwise a descriptive error message.
            """
        raw_columns = raw_df.columns.tolist() if raw_df is not None else []
        for col in cols:
            if col is None:
                return "❌ One of the columns could not be matched."
            if col not in cleaned_df.columns:
                if raw_df is not None and col in raw_columns:
                    return f"⚠️ Column '{col}' was removed during data cleaning."
                return f"❌ Column '{col}' does not exist in the dataset."
        return None

    async def eda_analysis(self, user_query, cleaned_df):
        """Run EDA on two columns detected from the query.

            Detects/validates columns, executes statistical tests and plots
            (e.g., posterior predictive checks, MCMC traces, ACF), and streams an
            interpreted summary via ``eda_agent``.

            Args:
                user_query: Natural language request (must imply ≥2 columns).
                cleaned_df: EDA-ready DataFrame.

            Yields:
                Markdown/text chunks summarizing EDA results and visuals.
            """
        detected_columns = self.extract_columns_from_query(user_query, cleaned_df.columns.tolist())

        if len(detected_columns) < 2:
            yield "Please specify at least two valid columns."
            return

        validate_msg = self.validate_columns(detected_columns, cleaned_df, self.df_raw)
        if validate_msg:
            yield validate_msg
            return

        print("🧠 Detected columns:", detected_columns)
        col1, col2 = detected_columns[:2]
        print("➡️ Using columns:", col1, col2)

        edaAnalysis(cleaned_df, col1, col2)

        image_paths = ["services/AI/plots/pp_check.png", "services/AI/plots/mcmc_trace.png", "services/AI/plots/mcmc_acf.png"]
        predefined_titles = ["Posterior Predictive Check", "MCMC Trace Plot", "Autocorrelation Function (ACF)"]
        predefined_filenames = ["pp_check.png", "mcmc_trace.png", "mcmc_acf.png"]
        json_files = {
            "num_num": "services/AI/data/num_num.json",
            "num_cat": "services/AI/data/num_cat.json",
            "cat_cat": "services/AI/data/cat_cat.json"
        }

        run_test(image_paths, predefined_titles, predefined_filenames, "services/AI/plots/scientific_image_analysis.json", "eda")

        try:
            async for chunk in self.eda_agent(json_files, cleaned_df, col1, col2, user_query):
                yield chunk
        except Exception as e:
            print("❌ EDA Agent Exception:", str(e))
            yield "EDA analysis failed due to internal error."

    @staticmethod
    def extract_shap_target_column(query: str, available_columns: list[str]) -> str:
        query = query.lower()
        for col in available_columns:
            if col.lower() in query:
                return col
        return None

    async def shap_analysis(self, user_query, cleaned_df):
        """Compute and explain SHAP values for a detected target variable.

            Trains the best-fit model for the target and streams SHAP interpretation
            via ``shap_agent``.

            Args:
                user_query: Query mentioning a target metric (e.g., ``"CTR"``).
                cleaned_df: Cleaned DataFrame used for modeling.

            Yields:
                Narrative chunks covering global/local importance and caveats.
            """
        available_columns = cleaned_df.columns.tolist()
        target_var = self.extract_shap_target_column(user_query, available_columns)

        if not target_var:
            yield "❌ Could not detect a target variable for SHAP analysis. Try asking about a specific variable like 'CTR' or 'conversions'."
            return

        validate_msg = self.validate_columns([target_var], cleaned_df, self.df_raw)
        if validate_msg:
            yield validate_msg
            return

        try:
            getBestModel(cleaned_df, target_var, query=user_query)
        except Exception as e:
            print("❌ Error in SHAP model training:", str(e))
            yield f"❌ Failed to compute SHAP analysis: {e}"
            return

        try:
            async for chunk in self.shap_agent(target_var, user_query):
                yield chunk
        except Exception as e:
            print("❌ Shap Agent Exception:", str(e))
            yield "Shap analysis failed due to internal error."

    async def ts_analysis(self, user_query, cleaned_df):
        """Perform time-series analysis (Prophet forecasts / Granger tests).

            Expands ranges to daily level, infers forecast horizon and metric(s),
            runs forecasts and/or Granger causality, optionally builds a dashboard,
            and streams explanations via ``ts_agent``.

            Args:
                user_query: Forecast/causality question (may include horizon like “next 30 days”).
                cleaned_df: Time-series-ready DataFrame.

            Yields:
                Stepwise summaries and final conclusions with saved artifacts.
            """
        expander = DailyDistributionExpander()
        metrics_to_expand = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        dashboard=None
        daily_df = expander.expand(
            cleaned_df,
            metrics_to_expand,
            start_col='date_start',
            end_col='date_stop',
            id_col='campaign_name'
        )

        daily_df.columns = [
            col.replace("daily_", "") if col.startswith("daily_") else col
            for col in daily_df.columns
        ]

        query = user_query.lower()
        found_cols = [col for col in metrics_to_expand if col.lower() in query]
        forecast_horizon = 30

        day_match = re.search(r"(\d+)\s*(g[üu]n|day[s]?)", query)
        company = None
        for c in daily_df["campaign_name"].unique().tolist():
            if re.search(rf"\b{re.escape(c.lower())}\b", query.lower()):
                company = c
                break
        if day_match:
            forecast_horizon = int(day_match.group(1))
        if len(found_cols) == 0:
            process = {"mode": "general", "forecast": False, "granger": False, "columns": [], "horizon": forecast_horizon}

        elif len(found_cols) == 1:
            process= {"mode": "single_metric", "forecast": True, "granger": True, "columns": found_cols,
                    "horizon": forecast_horizon}

        elif len(found_cols) == 2:
            process= {"mode": "pairwise_granger", "forecast": False, "granger": True, "columns": found_cols,
                    "horizon": None}

        else:
            process= {"mode": "multi_column_ambiguous", "forecast": False, "granger": False, "columns": found_cols,
                    "horizon": None}

        analyzer = TimeSeriesAnalyzer(df=daily_df, company=company, country="TR")

        if process["mode"] == "general":
            all_columns = metrics_to_expand
            all_results = []

            for col in all_columns:
                result = analyzer.run_prophet(col=col, horizon=process["horizon"])
                result["metric"] = col
                all_results.append(result)

            with open(f"services/AI/data/ts_metrics.json", "w") as f:
                json.dump(all_results, f, indent=4)


        elif process["mode"] == "single_metric" and len(process["columns"]) == 1:
            col = process["columns"][0]

            forecast_result = analyzer.run_prophet(col=col, horizon=process["horizon"])

            granger_result = analyzer.run_granger(col1=col, col2=None)

            dashboard = analyzer.plot_forecast_dashboard()

            combined_result = {
                "metric": col,
                "forecast_result": forecast_result,
                "granger_result": granger_result
            }
            output_path = f"services/AI/data/ts_metrics.json"
            with open(output_path, "w") as f:
                json.dump(combined_result, f, indent=4)

            # dashboard.servable("Forecast Dashboard")


        elif process["mode"] == "pairwise_granger" and len(process["columns"]) == 2:
            col1, col2 = process["columns"]

            granger_result = analyzer.run_granger(col1=col1, col2=col2)

            result_data = {
                "mode": "pairwise_granger",
                "col1": col1,
                "col2": col2,
                "granger_result": granger_result
            }

            filename = f"services/AI/data/ts_metrics.json"

            with open(filename, "w") as f:
                json.dump(result_data, f, indent=4)

            print(f"Granger result saved to {filename}")


        else:
            print("No valid processing mode matched.")

        try:
            async for chunk in self.ts_agent(user_query, dashboard):
                yield chunk
        except Exception as e:
            print("❌ Time Series Agent Exception:", str(e))
            yield "Time Series analysis failed due to internal error."


    def parse_geolift_query(self, user_query: str):
        import re

        patterns = [
            r"using (\w+) as metric.*?(\w+) as location.*?(\w+) as date",
            r"metric\s*[:=]\s*(\w+).*?location\s*[:=]\s*(\w+).*?date\s*[:=]\s*(\w+)",
            r"(\w+)\s+for\s+metric.*?(\w+)\s+for\s+location.*?(\w+)\s+for\s+date",
            r"metric is (\w+).*?location is (\w+).*?date is (\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                groups = match.groups()
                return {
                    "metric": groups[0],
                    "location": groups[1],
                    "date": groups[2]
                }
        return None

    def geo_analysis(self, user_query, df):
        """Run GeoLift power/impact analysis from a structured query.

            Parses metric/location/date, optional hyperparameters (budget, cpic, alpha,
            treatment start/duration), renames columns to GeoLift schema, executes
            ``powerAnalysis``, and returns an interpreted summary via ``geo_agent``.

            Args:
                user_query: Structured query (e.g., ``Using revenue as metric ...``).
                df: DataFrame containing metric, location, and date columns.

            Returns:
                Natural-language explanation of the GeoLift setup and results.
            """
        import os
        import re

        budget_value = 10000
        cpic = None
        alpha = 0.05
        treatment_start = None
        treatment_duration = None
        #df = pd.read_csv("geo_data.csv")
        #raw_columns = df.columns.tolist()
        raw_columns = self.df_raw.columns.tolist()
        available_columns = df.columns.tolist()
        removed = []

        # 🧠 Kolonları query'den çıkarmaya çalış
        parsed = self.parse_geolift_query(user_query)
        if not parsed:
            return "❌ Could not extract metric, location, and date columns from your query. Use formats like: 'Using spend as metric and region as location and date as date'."

        metric_col = self.find_best_column_match(parsed["metric"], available_columns)
        location_col = self.find_best_column_match(parsed["location"], available_columns)
        date_col = self.find_best_column_match(parsed["date"], available_columns)

        # ✅ Kolonların mevcudiyetini kontrol et
        for col_name, var in [("Metric", metric_col), ("Location", location_col), ("Date", date_col)]:
            if not var:
                removed.append(f"❌ {col_name} column not specified or matched in the dataset.")
            elif var not in raw_columns:
                removed.append(f"❌ {col_name} column '{var}' does not exist in the dataset.")
            elif var not in available_columns:
                removed.append(f"⚠️ {col_name} column '{var}' was removed during data cleaning.")

        if any(msg.startswith("❌") or msg.startswith("⚠️ Metric") for msg in removed):
            return "\n".join(removed) + "\n❌ Cannot proceed with GeoLift analysis."

        if match := re.search(r"budget (\d+(?:\.\d+)?)", user_query, re.IGNORECASE):
            budget_value = float(match.group(1))

        if match := re.search(r"cpic (\d+(?:\.\d+)?)", user_query, re.IGNORECASE):
            cpic = float(match.group(1))

        if match := re.search(r"alpha (\d+(?:\.\d+)?)", user_query, re.IGNORECASE):
            alpha = float(match.group(1))

        if match := re.search(r"treatment start (\d{4}-\d{2}-\d{2})", user_query, re.IGNORECASE):
            treatment_start = match.group(1)

        if match := re.search(r"treatment duration (\d+)", user_query, re.IGNORECASE):
            treatment_duration = int(match.group(1))

        # 🔄 GeoLift için rename et
        df = df.rename(columns={
            date_col: "time",
            location_col: "location",
            metric_col: "Y"
        })

        os.makedirs("services/AI/data", exist_ok=True)

        powerAnalysis(
            df=df,
            date_id="time",
            location_id="location",
            y_id="Y",
            budget=budget_value,
            cpic=cpic,
            treatment_start=treatment_start,
            treatment_duration=treatment_duration,
            alpha=alpha
        )

        return self.geo_agent("services/AI/data/model_summaries.json", user_query)

    @staticmethod
    def parse_causal_query(user_query: str):
        import re

        patterns = [
            # 1. Original explicit format
            r"Using (\w+) as treatment and (\w+) as outcome,? analyze the causal effect(?: with confounders \[(.*?)\])?(?: and instrument (\w+))?(?: at time (\d+))?(?: with target unit (\w+))?",

            # 2. Simpler natural language
            r"effect of (\w+) on (\w+)(?: with confounders \[(.*?)\])?(?: instrument (\w+))?(?: time (\d+))?(?: target unit (\w+))?",

            # 3. Very relaxed key=value style
            r"treatment\s*=\s*(\w+).*?outcome\s*=\s*(\w+)(?:.*?confounders\s*=\s*\[(.*?)\])?(?:.*?instrument\s*=\s*(\w+))?(?:.*?time\s*=\s*(\d+))?(?:.*?target\s*unit\s*=\s*(\w+))?"
        ]

        for pattern in patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                groups = match.groups()
                return {
                    "treatment": groups[0],
                    "outcome": groups[1],
                    "confounders": [x.strip() for x in groups[2].split(",")] if groups[2] else [],
                    "instrument": groups[3] if len(groups) > 3 else None,
                    "treatment_time": int(groups[4]) if len(groups) > 4 and groups[4] else None,
                    "target_unit": groups[5] if len(groups) > 5 else None
                }

        return None

    @staticmethod
    def parse_causalpy_query(user_query: str):
        import re

        patterns = [
            # 1. Net format: Using outcome with predictors
            r"Using (\w+) as outcome.*?with predictors \[(.*?)\](?:.*?at time (\d+))?",

            # 2. Key=value format
            r"outcome\s*=\s*(\w+).*?predictors\s*=\s*\[(.*?)\](?:.*?time\s*=\s*(\d+))?",

            # 3. Basit doğal dil
            r"effect on (\w+) with predictors \[(.*?)\](?: time (\d+))?"
        ]

        for pattern in patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                groups = match.groups()
                return {
                    "outcome": groups[0],
                    "predictors": [x.strip() for x in groups[1].split(",")] if groups[1] else [],
                    "treatment_time": int(groups[2]) if len(groups) > 2 and groups[2] else None
                }

        return None

    def dowhy_econml_analysis(self, user_query, cleaned_df):
        """Estimate causal effects with DoWhy/EconML and interpret results.

            Parses treatment/outcome/confounders (and optional instrument, treatment_time,
            target_unit), validates variables, runs ``causalModel``, generates diagnostics,
            and summarizes results via ``dowhy_agent``.

            Args:
                user_query: Causal query (key-value format).
                cleaned_df: Cleaned DataFrame suitable for causal estimation.

            Returns:
                Final narrative; prepended with warnings if variables were missing/removed.
            """
        import os

        parsed = self.parse_causal_query(user_query)

        if not parsed:
            return "❌ Could not parse the query format..."

        treatment_var = parsed["treatment"]
        outcome_var = parsed["outcome"]
        confounders_var = parsed["confounders"]
        instrument_var = parsed.get("instrument")
        treatment_time_var = parsed.get("treatment_time")
        target_unit_var = parsed.get("target_unit") or treatment_var

        raw_columns = self.df_raw.columns.tolist()
        available_columns = cleaned_df.columns.tolist()
        removed = []

        # 1. treatment
        if treatment_var not in raw_columns:
            removed.append(f"❌ Treatment column '{treatment_var}' does not exist in the dataset.")
        elif treatment_var not in available_columns:
            removed.append(f"⚠️ Treatment column '{treatment_var}' was removed due to high VIF.")

        # 2. outcome
        if outcome_var not in raw_columns:
            removed.append(f"❌ Outcome column '{outcome_var}' does not exist in the dataset.")
        elif outcome_var not in available_columns:
            removed.append(f"⚠️ Outcome column '{outcome_var}' was removed due to high VIF.")

        # 3. instrument
        if instrument_var:
            if instrument_var not in raw_columns:
                removed.append(f"❌ Instrument column '{instrument_var}' does not exist in the dataset.")
                instrument_var = None
            elif instrument_var not in available_columns:
                removed.append(f"⚠️ Instrument column '{instrument_var}' was removed due to high VIF.")
                instrument_var = None

        # 4. target_unit
        if target_unit_var not in raw_columns:
            removed.append(f"❌ Target unit column '{target_unit_var}' does not exist in the dataset.")
            target_unit_var = treatment_var
        elif target_unit_var not in available_columns:
            removed.append(f"⚠️ Target unit column '{target_unit_var}' was removed due to high VIF.")
            target_unit_var = treatment_var

        # 5. confounders
        valid_confounders = []
        removed_confounders = []
        missing_confounders = []

        for c in confounders_var:
            if c not in raw_columns:
                missing_confounders.append(c)
            elif c not in available_columns:
                removed_confounders.append(c)
            else:
                valid_confounders.append(c)

        if missing_confounders:
            removed.append(f"❌ These confounders do not exist in the dataset: {missing_confounders}")
        if removed_confounders:
            removed.append(f"⚠️ These confounders were removed due to high VIF: {removed_confounders}")

        confounders_var = valid_confounders

        # 6. Kritik eksikler varsa işlemi iptal et
        if any(msg.startswith("❌") or msg.startswith("⚠️ Treatment") or msg.startswith("⚠️ Outcome") for msg in
               removed):
            return "\n".join(removed) + "\n❌ Cannot proceed with analysis due to missing key variables."

        # 4. Kolon isimlerini normalize et
        treatment_var = self.find_best_column_match(treatment_var, available_columns)
        outcome_var = self.find_best_column_match(outcome_var, available_columns)
        if instrument_var:
            instrument_var = self.find_best_column_match(instrument_var, available_columns)
        confounders_var = [self.find_best_column_match(c, available_columns) for c in confounders_var]
        target_unit_var = self.find_best_column_match(target_unit_var, available_columns)

        # 5. Son kontrol
        validate_msg = self.validate_columns(
            [treatment_var, outcome_var, target_unit_var] +
            ([instrument_var] if instrument_var else []) +
            confounders_var,
            cleaned_df,
            self.df_raw
        )
        if validate_msg:
            return validate_msg

        # 6. Analizi çalıştır
        os.makedirs("services/AI/data", exist_ok=True)
        os.makedirs("services/AI/plots", exist_ok=True)

        causalModel(
            cleaned_df,
            treatment=treatment_var,
            outcome=outcome_var,
            confounders=confounders_var,
            instrument=instrument_var,
            treatment_time=treatment_time_var,
            target_unit=target_unit_var
        )

        image_paths = ["services/AI/plots/unobserved_confounder_heatmap.png"]
        predefined_titles = ["DoWhy Unobserved Confounder Heatmap"]
        predefined_filenames = ["unobserved_confounder_heatmap.png"]
        json_file = "services/AI/data/dowhy_econml_causal.json"

        run_test(image_paths, predefined_titles, predefined_filenames, "services/AI/plots/dowhy_image_analysis.json",
                 "dowhy")
        dowhy_result = self.dowhy_agent(json_file, user_query)

        # 7. Uyarıları analizin başına ekleyerek döndür
        if removed:
            return "\n".join(removed) + "\n\n✅ Analysis completed:\n" + dowhy_result
        else:
            return dowhy_result

    def causalPy_analysis(self, user_query, cleaned_df):
        """Run CausalPy synthetic control (Sklearn and Bayesian) and explain impact.

            Parses outcome/predictors/treatment_time, validates variables, executes
            ``synthetic_control`` variants, saves plots/JSON, and summarizes via
            ``causalpy_agent``.

            Args:
                user_query: CausalPy query (e.g., ``Using revenue as outcome, ...``).
                cleaned_df: Cleaned DataFrame.

            Returns:
                Narrative summary; warnings (if any) are prepended to the final text.
            """
        parsed = self.parse_causalpy_query(user_query)  # ✅ Aynı regex yapısı kullanılır

        if not parsed:
            return "❌ Could not parse the query format for CausalPy. Use formats like: 'Using X as outcome, analyze the causal effect with predictors [A, B, C] at time 10'."

        outcome_var = parsed["outcome"]
        predictors_var = parsed["predictors"]
        treatment_time_var = parsed.get("treatment_time")

        raw_columns = self.df_raw.columns.tolist()
        available_columns = cleaned_df.columns.tolist()
        removed = []

        # Outcome
        if outcome_var not in raw_columns:
            removed.append(f"❌ Outcome column '{outcome_var}' does not exist in the dataset.")
        elif outcome_var not in available_columns:
            removed.append(f"⚠️ Outcome column '{outcome_var}' was removed due to high VIF.")

        # Predictors
        valid_predictors = []
        removed_predictors = []
        missing_predictors = []

        for c in predictors_var:
            if c not in raw_columns:
                missing_predictors.append(c)
            elif c not in available_columns:
                removed_predictors.append(c)
            else:
                valid_predictors.append(c)

        if missing_predictors:
            removed.append(f"❌ These predictors do not exist in the dataset: {missing_predictors}")
        if removed_predictors:
            removed.append(f"⚠️ These predictors were removed due to high VIF: {removed_predictors}")

        predictors_var = valid_predictors

        # Check if critical elements are missing
        if any(msg.startswith("❌") or msg.startswith("⚠️ Outcome") for msg in removed):
            return "\n".join(removed) + "\n❌ Cannot proceed with analysis due to missing key variables."

        # Match actual column names (fuzzy matching)
        outcome_var = self.find_best_column_match(outcome_var, available_columns)
        predictors_var = [self.find_best_column_match(c, available_columns) for c in predictors_var]

        validate_msg = self.validate_columns([outcome_var] + predictors_var, cleaned_df, self.df_raw)
        if validate_msg:
            return validate_msg

        # Run analysis
        os.makedirs("services/AI/data", exist_ok=True)
        os.makedirs("services/AI/plots", exist_ok=True)

        try:
            synthetic_control(
                df=cleaned_df,
                outcome=outcome_var,
                predictors=predictors_var,
                treatment_time=treatment_time_var
            )

            image_paths = [
                "services/AI/plots/synthetic_control_with_sklearn.png",
                "services/AI/plots/synthetic_control_with_bayesian.png"
            ]
            predefined_titles = [
                "Synthetic Control (Sklearn-Based)",
                "Bayesian Synthetic Control (PyMC)"
            ]
            predefined_filenames = [
                "synthetic_control_with_sklearn.png",
                "synthetic_control_with_bayesian.png"
            ]
            json_file = "services/AI/data/sklearn_and_bayesian_causal_impact_summary.json"

            if not os.path.exists(json_file):
                return "❌ Error: JSON file containing causal analysis results was not created."

            run_test(image_paths, predefined_titles, predefined_filenames,
                     "services/AI/plots/bayesian_sklearn_image_analysis.json", "causalpy")
            result = self.causalpy_agent(json_file, user_query)

        except Exception as e:
            result = f"❌ An error occurred during CausalPy analysis: {str(e)}"

        return "\n".join(removed) + "\n\n✅ Analysis completed:\n" + result if removed else result

    async def combined_analysis(self, query):
        """Route a user query to the appropriate analysis pipeline and stream outputs.

            Uses ``determine_query_type`` and clears previous state (non-wiki). Dispatches to
            EDA/SHAP/TS/GeoLift/DoWhy-EconML/CausalPy/AB/Assoc or general agent/search.

            Args:
                query: End-user question.

            Yields:
                Streaming results produced by the selected pipeline.
            """
        query_type = await self.determine_query_type(query)
        print(f"🧠 DEBUG: Determined Query Type → {query_type}")
        if query_type != "wiki":
            self.checkpointer.delete_thread(self.user_id)
        if query_type == "dowecon":
            print("🔍 Running **Causal Inference Analysis With DoWhy and EconML**...")
            dowhy_results = self.dowhy_econml_analysis(query, self.df_raw.copy())
            self.console.print(Markdown(dowhy_results))
            print("✅ Analysis results saved.")
            yield dowhy_results

        elif query_type == "causalpy":
            print("🔍 Running **Bayesian CausalPy Analysis**...")
            causalPy_res = self.causalPy_analysis(query, self.df_raw.copy())
            self.console.print(Markdown(causalPy_res))
            print("✅ Analysis results saved.")
            yield causalPy_res

        elif query_type == "eda":
            print("🔍 Running **EDA Analysis**...")
            async for chunk in self.eda_analysis(query, self.eda_cleaned_df):
                yield chunk

        elif query_type == "shap":
            print("🔍 Running **SHAP Analysis**...")
            async for chunk in self.shap_analysis(query, self.causal_cleaned_df):
                yield chunk

        elif query_type == "geolift":
            print("🔍 Running **GeoLift Analysis**...")
            geo_df = self.df_raw.copy().dropna()
            async for chunk in self.geo_analysis(query, geo_df):
                yield chunk

        elif query_type == "analyze":
            print("🔍 Running **General Agent Analysis**...")
            async for chunk in self.analyze_agent(query):
                yield chunk

        elif query_type == "wiki":
            print("🔍 Running **Wikipedia Searching**...")
            async for chunk in self.wiki_agent(query):
                yield chunk

        elif query_type == "ts":
            print("🔍 Running **Time Series Analysis**...")
            async for chunk in self.ts_analysis(query, self.ts_cleaned_df):
                yield chunk


        elif query_type == "ab":
            print("🔍 Running **A-B Testing Analysis**...")
            get_analysis(self.ts_cleaned_df, query)
            async for chunk in self.ab_agent(query):
                yield chunk

        elif query_type == "assoc":
            print("🔍 Running **Association Rules**...")
            run_assoc(self.ts_cleaned_df, query)
            async for chunk in self.assoc_agent(query):
                yield chunk
        else:
            yield "❌ Could not determine the analysis type. Please refine your query."

    async def combined_analysis_plots(self, query):
        """Generate visualizations when the query intent is plotting.

            Detects plotting intent, coerces datetime-like object columns, extracts target
            columns, forms pairwise combinations, and produces figures via ``generate_plots``.

            Args:
                query: Visualization request mentioning ≥2 columns.

            Returns:
                A list of figure objects or saved file paths (implementation-specific).
            """
        query_type = await self.determine_query_type(query)
        print(f"🧠 DEBUG: Determined Query Type → {query_type}")

        if query_type == "plot":
            print("🔍 Running **PLOT Analysis**...")

            plot_df = self.df_raw.dropna()
            datetime_cols = []
            for col in plot_df.select_dtypes(include=['object']).columns:
                try:
                    converted = pd.to_datetime(plot_df[col], errors='coerce')
                    if converted.notna().sum() / len(converted) > 0.7:
                        datetime_cols.append(col)
                        plot_df[col] = converted
                except Exception:
                    pass

            print(f"✅ Identified datetime columns: {datetime_cols}")
            available_columns = plot_df.columns.tolist()
            detected_columns = self.extract_columns_from_query(query, available_columns)

            if len(detected_columns) < 2:
                print("❌ Please specify at least two valid columns for visualization.")
                return

            column_combinations = list(combinations(detected_columns, 2))
            plots = self.generate_plots(plot_df, column_combinations)
            print("✅ Plot successfully generated!")
            return plots

