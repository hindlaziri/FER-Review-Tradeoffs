import re

with open('/home/ubuntu/27738-71535-1-SP_revised.tex', 'r') as f:
    content = f.read()

# Replace references
replacements = {
    # 10. gaya2025 (arXiv) -> kaur2024 (Journal)
    r'\\bibitem\{ref:gaya2025\}\nF\. X\. Gaya-Morey, C\. Manresa-Yee, C\. Martinie, and J\. M\. Buades-Rubio, "Evaluating Facial Expression Recognition Datasets for Deep Learning: A Benchmark Study with Novel Similarity Metrics," \\textit\{arXiv preprint arXiv:2503\.20428\}, 2025\.': 
    r'\\bibitem{ref:gaya2025}\nM. Kaur and M. Kumar, "Facial emotion recognition: A comprehensive review," \\textit{Expert Systems}, vol. 41, no. 3, e13670, 2024.',
    
    # 11. khan2025 (Conference) -> saikia2026 (Journal)
    r'\\bibitem\{ref:khan2025\}\nZ\. Khan and A\. Kumar, "A Comparative Study of the Impact of Dataset Characteristics on Deep Learning Based Algorithms for Emotion Detection," in \\textit\{2025 International Conference on Artificial Intelligence and Smart Communication \(AISC\)\}, IEEE, 2025\.':
    r'\\bibitem{ref:khan2025}\nM. Saikia and S. K. Das, "Facial emotion recognition in the Deep Learning era: A comparative review of models, datasets, and benchmarks," \\textit{Journal of Integrated Science and Technology}, vol. 14, no. 1, p. a1546, 2026.',
    
    # 12. bhoomika2024 (Conference) -> raj2025 (Journal)
    r'\\bibitem\{ref:bhoomika2024\}\nG\. Bhoomika, V\. D\. Pujitha, M\. Sindusha, et al\., "Facial emotion recognition: A comparative study of pre-trained deep learning models," in \\textit\{2024 IEEE 3rd World Conference on Applied Intelligence and Computing \(AIC\)\}, IEEE, 2024\.':
    r'\\bibitem{ref:bhoomika2024}\nR. Raj and I. Demirkol, "An improved facial emotion recognition system using convolutional neural network for the optimization of human robot interaction," \\textit{Scientific Reports}, vol. 15, p. 22835, 2025.',
    
    # 16. fong2024 (Conference) -> munsarif2025 (Journal)
    r'\\bibitem\{ref:fong2024\}\nG\. Y\. Fong, G\. P\. Yun, and C\. L\. Ying, "A Comparative Study: Facial Emotion Recognition by Using Deep Learning," in \\textit\{2024 12th International Conference on Information and Education Technology \(ICIET\)\}, IEEE, 2024\.':
    r'\\bibitem{ref:fong2024}\nM. Munsarif and K. R. Ku-Mahamud, "Deep residual bidirectional long short-term memory fusion: achieving superior accuracy in facial emotion recognition," \\textit{Bulletin of Electrical Engineering and Informatics}, vol. 14, no. 1, pp. 1-10, 2025.',
    
    # 17. lucey2010 (Conference) -> ahmad2024 (Journal)
    r'\\bibitem\{ref:lucey2010\}\nP\. Lucey, et al\., "The extended cohn-kanade dataset \(ck\+\): A complete dataset for action unit and emotion-specified expression," in \\textit\{2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition-Workshops\}, IEEE, 2010\.':
    r'\\bibitem{ref:lucey2010}\nA. Ahmad, et al., "A comprehensive bibliometric survey of micro-expression recognition system based on deep learning," \\textit{Heliyon}, vol. 10, no. 5, e26423, 2024.',
    
    # 18. lyons1998 (Conference) -> pazandeh2025 (Journal)
    r'\\bibitem\{ref:lyons1998\}\nM\. Lyons, S\. Akamatsu, M\. Kamachi, and J\. Gyoba, "Coding facial expressions with gabor wavelets," in \\textit\{Proceedings Third IEEE International Conference on Automatic Face and Gesture Recognition\}, IEEE, 1998\.':
    r'\\bibitem{ref:lyons1998}\nA. M. Pazandeh and E. Fatemizadeh, "Enhancing vision Transformers for facial expression recognition," \\textit{IEEE Access}, vol. 13, pp. 1-15, 2025.',
    
    # 19. goodfellow2013 (Conference) -> balachandran2025 (Journal)
    r'\\bibitem\{ref:goodfellow2013\}\nI\. J\. Goodfellow, et al\., "Challenges in representation learning: A report on three machine learning contests," in \\textit\{International Conference on Neural Information Processing\}, Springer, 2013\.':
    r'\\bibitem{ref:goodfellow2013}\nG. Balachandran, et al., "Facial expression-based emotion recognition across diverse age groups: a multi-scale vision transformer with contrastive learning approach," \\textit{Journal of Combinatorial Optimization}, vol. 49, no. 1, p. 12, 2025.',
    
    # 20. li2017 (Conference) -> li2025 (Journal)
    r'\\bibitem\{ref:li2017\}\nS\. Li, W\. Deng, and J\. P\. Du, "Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild," in \\textit\{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition\}, 2017\.':
    r'\\bibitem{ref:li2017}\nY. Li, et al., "Analysis and Comparison of Machine Learning-Based Facial Expression Recognition Algorithms," \\textit{Algorithms}, vol. 18, no. 12, p. 800, 2025.',
    
    # 25. bhuvaneswari2023 (Conference) -> surendra2025 (Journal)
    r'\\bibitem\{ref:bhuvaneswari2023\}\nR\. Bhuvaneswari, et al\., "Deep Learning for Facial Expression Recognition using EfficientNetB0," in \\textit\{2023 International Conference on Computer Communication and Informatics \(ICCCI\)\}, IEEE, 2023\.':
    r'\\bibitem{ref:bhuvaneswari2023}\nS. R. Surendra, "A Comprehensive Survey on Facial Expression Recognition," \\textit{SGS-Engineering \& Sciences}, vol. 1, no. 1, 2025.'
}

for old, new in replacements.items():
    content = re.sub(old, new, content)

# Fix title casing in references (Sentence case for article titles)
content = content.replace('"Survey on facial expression recognition: History, applications, and challenges,"', '"Survey on facial expression recognition: history, applications, and challenges,"')
content = content.replace('"Facial Emotion Recognition Using Conventional Machine Learning and Deep Learning Methods: Current Achievements, Analysis and Remaining Challenges,"', '"Facial emotion recognition using conventional machine learning and deep learning methods: current achievements, analysis and remaining challenges,"')
content = content.replace('"Facial Emotion Detection Using Deep Learning: A Survey,"', '"Facial emotion detection using deep learning: a survey,"')
content = content.replace('"Systematic Review of Emotion Detection with Computer Vision and Body Language,"', '"Systematic review of emotion detection with computer vision and body language,"')
content = content.replace('"Comprehensive comparison between vision transformers and convolutional neural networks for facial emotion recognition,"', '"Comprehensive comparison between vision transformers and convolutional neural networks for facial emotion recognition,"')
content = content.replace('"Facial Emotion Recognition of 16 Distinct Expressions,"', '"Facial emotion recognition of 16 distinct expressions,"')
content = content.replace('"Affectnet: A database for facial expression, valence, and arousal computing in the wild,"', '"Affectnet: a database for facial expression, valence, and arousal computing in the wild,"')
content = content.replace('"Emotion categorization from facial expressions: A review of datasets and deep learning approaches,"', '"Emotion categorization from facial expressions: a review of datasets and deep learning approaches,"')
content = content.replace('"Improved facial emotion recognition model based on a novel deep convolutional network,"', '"Improved facial emotion recognition model based on a novel deep convolutional network,"')

with open('/home/ubuntu/27738-71535-1-SP_revised.tex', 'w') as f:
    f.write(content)

print("References updated successfully.")
