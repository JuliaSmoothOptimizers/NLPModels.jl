for field in fieldnames(NLSMeta)
  meth = Symbol("get_", field)
  @eval begin
    @doc """
        $($meth)(nls)
        $($meth)(nls_meta)
    Return the value $($(QuoteNode(field))) from nls\\_meta or nls.nls\\_meta.
    """
    $meth(nls_meta::NLSMeta) = getproperty(nls_meta, $(QuoteNode(field)))
  end
  @eval $meth(nls::AbstractNLSModel) = $meth(nls.nls_meta)
  @eval export $meth
end
