(ns pull-model
  (:require [babashka.process :as process]
            [clojure.core.async :as async]))

(try
  (let [llm (get (System/getenv) "LLM")
        url (get (System/getenv) "OLLAMA_BASE_URL")]
    (println (format "pulling ollama model %s using %s" llm url))
    (if (and llm
         url
         (not (some #(.startsWith llm %) ["gpt" 
                                          "claudev2"]))
         (not (some #(.startsWith llm %) ["ai21.jamba-instruct-v1:0"
                                          "amazon.titan"
                                          "anthropic.claude"
                                          "cohere.command"
                                          "meta.llama"
                                          "mistral.mi"])))

      ;; ----------------------------------------------------------------------
      ;; just call `ollama pull` here - create OLLAMA_HOST from OLLAMA_BASE_URL
      ;; ----------------------------------------------------------------------
      ;; TODO - this still doesn't show progress properly when run from docker compose

      (let [done (async/chan)]
        (async/go-loop [n 0]
          (let [[v _] (async/alts! [done (async/timeout 5000)])]
            (if (= :stop v) :stopped (do (println (format "... pulling model (%ss) - will take several minutes" (* n 10))) (recur (inc n))))))
        (process/shell {:env {"OLLAMA_HOST" url "HOME" (System/getProperty "user.home")} :out :inherit :err :inherit} (format "bash -c './bin/ollama show %s --modelfile > /dev/null || ./bin/ollama pull %s'" llm llm))
        (async/>!! done :stop))

      (println "OLLAMA model only pulled if both LLM and OLLAMA_BASE_URL are set and the LLM model is not gpt")))
  (catch Throwable _ (System/exit 1)))
