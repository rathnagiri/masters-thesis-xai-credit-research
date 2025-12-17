% Compliance rules (GDPR completeness, Basel risk flags, MiFID transparency).
% Facts expected from caller:
%   mentioned(atom).              % factors/terms mentioned in the explanation
%   required(atom).               % factors that must be present (e.g., income, debt, credit_history)
%   risk_flag(atom).              % risk factors that should be acknowledged if present (e.g., high_debt)
%   decision(atom).               % optional: approve / reject
%   require_transparency.         % optional flag to enforce transparency mention

% GDPR Art.22 completeness: all required terms must be mentioned.
violation(missing_required(T)) :- required(T), not mentioned(T).

% Basel risk awareness: if a risk_flag exists, it must be mentioned.
violation(risk_not_addressed(R)) :- risk_flag(R), not mentioned(R).

% MiFID transparency / clarity (only if required_transparency asserted).
violation(missing_transparency) :- require_transparency, not mentioned(transparency).

% If no violations, mark pass.
pass :- not violation(_).

#show violation/1.
#show pass/0.
