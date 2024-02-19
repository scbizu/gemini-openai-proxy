package api

import (
	"crypto/tls"
	"fmt"
	"net/http"
	"net/url"
)

type CustomizedTransport struct {
	apiKey      string
	tlsProxyURL string
}

func (c *CustomizedTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	tp := http.DefaultTransport
	if c.tlsProxyURL == "" {
		return tp.RoundTrip(req)
	}
	p, ok := tp.(*http.Transport)
	if !ok {
		return nil, fmt.Errorf("http.DefaultTransport is not an *http.Transport")
	}
	u, err := url.Parse(c.tlsProxyURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse tlsProxyURL: %w", err)
	}
	p.Proxy = http.ProxyURL(u)
	p.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	// clone the origin request
	cloneReq := new(http.Request)
	*cloneReq = *req
	args := req.URL.Query()
	args.Set("key", c.apiKey)
	cloneReq.URL.RawQuery = args.Encode()
	resp, err := tp.RoundTrip(cloneReq)
	if err != nil {
		return nil, fmt.Errorf("failed to round trip the request: %w", err)
	}
	return resp, nil
}
