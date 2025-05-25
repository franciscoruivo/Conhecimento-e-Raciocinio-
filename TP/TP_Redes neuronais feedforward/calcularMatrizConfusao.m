function matriz_conf = calcularMatrizConfusao(targets, outputs)
% CALCULARMATRIZCONFUSAO - Calcula a matriz de confusão para os resultados da rede
% 
% Parâmetros:
%   targets - Matriz de targets (codificação one-hot)
%   outputs - Saídas da rede neural
%
% Retorna:
%   matriz_conf - Matriz de confusão em percentagem

    % Número de classes
    num_classes = size(targets, 1);
    
    % Obter classes preditas e reais
    [~, pred] = max(outputs);
    [~, real] = max(targets);
    
    % Inicializar matriz de confusão
    matriz_conf = zeros(num_classes, num_classes);
    
    % Calcular contagens por classe
    contagem_por_classe = zeros(num_classes, 1);
    
    % Preencher matriz de confusão
    for i = 1:length(real)
        classe_real = real(i);
        classe_pred = pred(i);
        
        matriz_conf(classe_real, classe_pred) = matriz_conf(classe_real, classe_pred) + 1;
        contagem_por_classe(classe_real) = contagem_por_classe(classe_real) + 1;
    end
    
    % Converter para percentagem
    for i = 1:num_classes
        if contagem_por_classe(i) > 0
            matriz_conf(i, :) = (matriz_conf(i, :) / contagem_por_classe(i)) * 100;
        end
    end
end 